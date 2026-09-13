#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_rwpe_node_t.py

为 DyGFormerD 按 (node_id, t) 预热 RWPE（正样本两端）。

对数据中每条交互 (src, dst, t)：
  计算并存储 key=(src, t) 与 (dst, t) 的序列级 RWPE（同一 key 只算一次）。
训练时负样本的 (neg_dst, t) 可走模型内懒缓存（同一算子）。

存储格式（chunk dict）:
  key:   (int node_id, float t)
  value: np.float16 array, shape (L, rwpe_dim)

Usage:
  python3 precompute_rwpe_node_t.py --dataset_name enron
  python3 precompute_rwpe_node_t.py --dataset_name wikipedia --num_workers 8
"""

import os
import json
import time
import argparse
import pickle
import math
import multiprocessing as mp

import numpy as np
import pandas as pd

from precompute_rwpe import (
    build_neighbor_structure,
    build_center_sequence,
    compute_rwpe_for_sequence,
)


_G_node_nbr_ids = None
_G_node_nbr_times = None
_G_args = None


def _init_worker(nbr_ids, nbr_times, args):
    global _G_node_nbr_ids, _G_node_nbr_times, _G_args
    _G_node_nbr_ids = nbr_ids
    _G_node_nbr_times = nbr_times
    _G_args = args


def _process_key_chunk(task):
    chunk_id, keys, out_dir, overwrite = task
    chunk_path = os.path.join(out_dir, f'chunk_{chunk_id:05d}.pkl')
    if os.path.exists(chunk_path) and not overwrite:
        print(f'  Chunk {chunk_id:05d} already exists, skipping.', flush=True)
        return chunk_id, 0, 0

    args = _G_args
    chunk_data = {}
    n_ok = n_err = 0
    for nid, t in keys:
        k = (int(nid), float(t))
        try:
            seq = build_center_sequence(
                int(nid), float(t),
                _G_node_nbr_ids, _G_node_nbr_times,
                args.max_input_sequence_length,
            )
            pe = compute_rwpe_for_sequence(
                seq, float(t),
                _G_node_nbr_ids, _G_node_nbr_times,
                args.rwpe_dim, args.max_subgraph_size,
            )
            chunk_data[k] = pe  # float16 (L, dim) or zeros
            n_ok += 1
        except Exception as exc:
            chunk_data[k] = np.zeros((1, args.rwpe_dim), dtype=np.float16)
            n_err += 1
            if getattr(args, 'verbose', False):
                print(f'    [ERROR] chunk={chunk_id} key={k}: {exc}', flush=True)

    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk_data, f, protocol=4)

    print(
        f'  → Chunk {chunk_id:05d} done: {len(keys)} keys  ok={n_ok} error={n_err}',
        flush=True,
    )
    return chunk_id, n_ok, n_err


def collect_unique_node_t_keys(src_ids, dst_ids, timestamps):
    """Unique (node_id, t) from positive endpoints, stable order."""
    seen = set()
    keys = []
    for src, dst, t in zip(src_ids.tolist(), dst_ids.tolist(), timestamps.tolist()):
        tf = float(t)
        for nid in (int(src), int(dst)):
            k = (nid, tf)
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def main(args):
    t_wall = time.time()
    csv_path = os.path.join('processed_data', args.dataset_name, f'ml_{args.dataset_name}.csv')
    print(f'Loading {csv_path} ...')
    graph_df = pd.read_csv(csv_path)

    src_ids = graph_df.u.values.astype(np.int64)
    dst_ids = graph_df.i.values.astype(np.int64)
    timestamps = graph_df.ts.values.astype(np.float64)
    N = len(src_ids)
    print(f'  {N} interactions, nodes up to {max(src_ids.max(), dst_ids.max())}')

    print('Building neighbor structure ...')
    node_nbr_ids, node_nbr_times = build_neighbor_structure(src_ids, dst_ids, timestamps)
    print(f'  Done. {len(node_nbr_ids)} node slots.')

    print('Collecting unique (node_id, t) keys from positive endpoints ...')
    keys = collect_unique_node_t_keys(src_ids, dst_ids, timestamps)
    K = len(keys)
    print(f'  {K} unique keys (from {2 * N} endpoint slots)')

    tag = f'seq{args.max_input_sequence_length}_dim{args.rwpe_dim}_ms{args.max_subgraph_size}'
    out_dir = os.path.join('processed_data', args.dataset_name, f'rwpe_node_t_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'lazy_cache'), exist_ok=True)
    print(f'Output → {out_dir}')

    node_arr = np.array([k[0] for k in keys], dtype=np.int64)
    time_arr = np.array([k[1] for k in keys], dtype=np.float64)
    np.savez(os.path.join(out_dir, 'keys.npz'), node_ids=node_arr, times=time_arr)

    num_chunks = math.ceil(K / args.chunk_size) if K > 0 else 0
    meta = {
        'dataset_name': args.dataset_name,
        'key_type': 'node_t',
        'max_input_sequence_length': args.max_input_sequence_length,
        'rwpe_dim': args.rwpe_dim,
        'max_subgraph_size': args.max_subgraph_size,
        'chunk_size': args.chunk_size,
        'num_interactions': N,
        'num_keys': K,
        'num_chunks': num_chunks,
        'time_edge_rule': 'neighbor_times <= current_time',
        'neighbor_cutoff': 'times < t (searchsorted left)',
        'key_encoding': 'tuple(int node_id, float64 t) exact from csv',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    if K == 0:
        print('No keys to process.')
        return

    tasks = []
    for chunk_id in range(num_chunks):
        lo = chunk_id * args.chunk_size
        hi = min(lo + args.chunk_size, K)
        tasks.append((chunk_id, keys[lo:hi], out_dir, args.overwrite))

    num_workers = min(args.num_workers, num_chunks)
    n_ok = n_err = 0
    print(f'Processing {num_chunks} chunks with {num_workers} workers ...')
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(node_nbr_ids, node_nbr_times, args),
    ) as pool:
        for _, ok, err in pool.imap_unordered(_process_key_chunk, tasks):
            n_ok += ok
            n_err += err

    elapsed = time.time() - t_wall
    print(f'\nFinished. keys={K} ok={n_ok} error={n_err}')
    print(f'Total time: {elapsed / 60:.1f} min  |  Output: {out_dir}')
    print('Train with: --rwpe_precomputed_dir', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute RWPE keyed by (node_id, t) for DyGFormerD.'
    )
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--max_input_sequence_length', type=int, default=512)
    parser.add_argument('--rwpe_dim', type=int, default=8)
    parser.add_argument('--max_subgraph_size', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
