#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_rwpe_node_t_full.py

为 DyGFormerD 按 (node_id, t) 全覆盖预计算 RWPE，使随机负样本也能查表命中。

对每个查询时刻 t、候选集中每个节点 u，计算并存储 PE(u, t)。
候选集默认与 NegativeEdgeSampler.random_sample 一致：unique(dst)。

存储格式与 precompute_rwpe_node_t.py 相同：
  key:   (int node_id, float t)
  value: np.float16 array, shape (L, rwpe_dim)

输出目录:
  processed_data/<ds>/rwpe_node_t_full_seq{S}_dim{D}_ms{M}/

Usage:
  # 先估算规模（强烈建议）
  python3 precompute_rwpe_node_t_full.py --dataset_name uci --estimate_only

  # 正式跑（超 max_keys 会拒绝，除非 --force）
  python3 precompute_rwpe_node_t_full.py --dataset_name uci --num_workers 8

  # 冒烟：只算前 N 个 key
  python3 precompute_rwpe_node_t_full.py --dataset_name uci --max_keys 10000 --force
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
    chunk_id, node_ids, times, out_dir, overwrite = task
    chunk_path = os.path.join(out_dir, f'chunk_{chunk_id:05d}.pkl')
    if os.path.exists(chunk_path) and not overwrite:
        print(f'  Chunk {chunk_id:05d} already exists, skipping.', flush=True)
        return chunk_id, 0, 0

    args = _G_args
    chunk_data = {}
    n_ok = n_err = 0
    for nid, t in zip(node_ids.tolist(), times.tolist()):
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
            chunk_data[k] = pe
            n_ok += 1
        except Exception as exc:
            chunk_data[k] = np.zeros((1, args.rwpe_dim), dtype=np.float16)
            n_err += 1
            if getattr(args, 'verbose', False):
                print(f'    [ERROR] chunk={chunk_id} key={k}: {exc}', flush=True)

    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk_data, f, protocol=4)

    print(
        f'  → Chunk {chunk_id:05d} done: {len(node_ids)} keys  ok={n_ok} error={n_err}',
        flush=True,
    )
    return chunk_id, n_ok, n_err


def get_candidate_nodes(src_ids, dst_ids, candidate):
    if candidate == 'unique_dst':
        return np.unique(dst_ids.astype(np.int64))
    if candidate == 'unique_src':
        return np.unique(src_ids.astype(np.int64))
    if candidate == 'unique_all':
        return np.unique(np.concatenate([src_ids, dst_ids]).astype(np.int64))
    raise ValueError(f'Unknown candidate={candidate}')


def get_query_times(timestamps, time_mode):
    ts = timestamps.astype(np.float64)
    if time_mode == 'per_edge':
        # Keep edge order; duplicate timestamps remain (product will dedup via unique times
        # only if unique_ts). For per_edge we use every edge time but unique(ts) is almost
        # the same when timestamps are nearly unique; we still expand over unique(ts)
        # within each mode for stable key arrays.
        # per_edge: one row per edge timestamp (may duplicate identical floats)
        return ts.copy()
    if time_mode == 'unique_ts':
        return np.unique(ts)
    raise ValueError(f'Unknown time_mode={time_mode}')


def build_key_arrays(cand_nodes, query_times, max_keys=None):
    """
    Cartesian product: for each t in query_times, for each u in cand_nodes -> (u, t).
    Returns node_ids, times as int64/float64 arrays (possibly truncated by max_keys).
    Memory: O(K) with K = |cand| * |times|.
    """
    n_c = len(cand_nodes)
    n_t = len(query_times)
    K = int(n_c) * int(n_t)
    if max_keys is not None and K > max_keys:
        # Truncate by taking first ceil(max_keys / n_c) times (stable prefix)
        n_t_keep = max(1, int(math.ceil(max_keys / float(n_c))))
        query_times = query_times[:n_t_keep]
        n_t = len(query_times)
        K = n_c * n_t
        if K > max_keys:
            # also truncate candidates on last incomplete time block
            n_full = max_keys // n_c
            rem = max_keys - n_full * n_c
            if n_full == 0:
                node_ids = np.tile(cand_nodes[:max_keys], 1)
                times = np.repeat(query_times[:1], max_keys)
                return node_ids.astype(np.int64), times.astype(np.float64), max_keys
            node_ids = np.concatenate([
                np.tile(cand_nodes, n_full),
                cand_nodes[:rem],
            ])
            times = np.concatenate([
                np.repeat(query_times[:n_full], n_c),
                np.repeat(query_times[n_full:n_full + 1], rem),
            ])
            return node_ids.astype(np.int64), times.astype(np.float64), max_keys

    # tile: for each t, all candidates
    node_ids = np.tile(cand_nodes, n_t)
    times = np.repeat(query_times, n_c)
    return node_ids.astype(np.int64), times.astype(np.float64), K


def estimate_disk_gb(num_keys, bytes_per_key=10000):
    """Rough estimate from uci positive-only (~1.2G / 119404 ≈ 10KB/key)."""
    return num_keys * bytes_per_key / (1024.0 ** 3)


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

    cand = get_candidate_nodes(src_ids, dst_ids, args.candidate)
    query_times_full = get_query_times(timestamps, args.time_mode)
    # For key uniqueness: if per_edge has duplicate timestamps, product would duplicate
    # identical (u,t). Dedup times for the product while keeping meta honest.
    if args.time_mode == 'per_edge':
        query_times = np.unique(query_times_full)
        print(
            f'  time_mode=per_edge: {len(query_times_full)} edge times → '
            f'{len(query_times)} unique timestamps for product'
        )
    else:
        query_times = query_times_full

    K_full = int(len(cand)) * int(len(query_times))
    disk_gb = estimate_disk_gb(K_full, args.bytes_per_key_est)
    print(f'Candidate: {args.candidate} → |V_cand|={len(cand)}')
    print(f'Time mode: {args.time_mode} → |T|={len(query_times)}')
    print(f'Full keys (cartesian): {K_full:,}')
    print(f'Disk estimate (~{args.bytes_per_key_est / 1024:.0f}KB/key): ~{disk_gb:.1f} GB')
    print(f'Chunks if chunk_size={args.chunk_size}: {math.ceil(K_full / args.chunk_size):,}')

    if args.estimate_only:
        print('\n[estimate_only] Exit without writing. Re-run without --estimate_only to compute.')
        if K_full > args.refuse_above_keys and not args.force:
            print(
                f'NOTE: K={K_full:,} > refuse_above_keys={args.refuse_above_keys:,}; '
                f'full run needs --force (or raise --refuse_above_keys).'
            )
        return

    if K_full > args.refuse_above_keys and not args.force and args.max_keys is None:
        raise SystemExit(
            f'Refusing to run: {K_full:,} keys > --refuse_above_keys={args.refuse_above_keys:,}.\n'
            f'  Use --estimate_only, or --max_keys N for a smoke test, or --force to override.\n'
            f'  Disk estimate ~{disk_gb:.1f} GB.'
        )

    print('Building neighbor structure ...')
    node_nbr_ids, node_nbr_times = build_neighbor_structure(src_ids, dst_ids, timestamps)
    print(f'  Done. {len(node_nbr_ids)} node slots.')

    print('Building key arrays ...')
    node_arr, time_arr, K = build_key_arrays(cand, query_times, max_keys=args.max_keys)
    if args.max_keys is not None and K < K_full:
        print(f'  Truncated by --max_keys: using {K:,} / {K_full:,} keys')
    else:
        print(f'  {K:,} keys')

    tag = f'seq{args.max_input_sequence_length}_dim{args.rwpe_dim}_ms{args.max_subgraph_size}'
    out_dir = os.path.join(
        'processed_data', args.dataset_name, f'rwpe_node_t_full_{tag}'
    )
    if args.out_suffix:
        out_dir = out_dir + '_' + args.out_suffix
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'lazy_cache'), exist_ok=True)
    print(f'Output → {out_dir}')

    np.savez(os.path.join(out_dir, 'keys.npz'), node_ids=node_arr, times=time_arr)

    num_chunks = math.ceil(K / args.chunk_size) if K > 0 else 0
    meta = {
        'dataset_name': args.dataset_name,
        'key_type': 'node_t',
        'coverage': 'full_cand_per_time',
        'candidate': args.candidate,
        'time_mode': args.time_mode,
        'num_cand': int(len(cand)),
        'num_query_times': int(len(query_times)),
        'max_input_sequence_length': args.max_input_sequence_length,
        'rwpe_dim': args.rwpe_dim,
        'max_subgraph_size': args.max_subgraph_size,
        'chunk_size': args.chunk_size,
        'num_interactions': N,
        'num_keys': K,
        'num_keys_full_cartesian': K_full,
        'truncated': bool(args.max_keys is not None and K < K_full),
        'num_chunks': num_chunks,
        'time_edge_rule': 'neighbor_times <= current_time',
        'neighbor_cutoff': 'times < t (searchsorted left)',
        'key_encoding': 'tuple(int node_id, float64 t) exact from csv',
        'note': 'For fair train/eval: lookup-only; miss should not use a different online path.',
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
        tasks.append((chunk_id, node_arr[lo:hi], time_arr[lo:hi], out_dir, args.overwrite))

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
    print('Recommend: strict lookup (no asymmetric online miss path) for fairness.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Full-coverage RWPE precompute keyed by (node_id, t) for DyGFormerD.'
    )
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument(
        '--candidate', type=str, default='unique_dst',
        choices=['unique_dst', 'unique_src', 'unique_all'],
        help='Candidate nodes; unique_dst matches NegativeEdgeSampler.random_sample dst space.',
    )
    parser.add_argument(
        '--time_mode', type=str, default='unique_ts',
        choices=['unique_ts', 'per_edge'],
        help='Query times: unique timestamps (default) or every edge time (then uniquified).',
    )
    parser.add_argument('--max_input_sequence_length', type=int, default=512)
    parser.add_argument('--rwpe_dim', type=int, default=8)
    parser.add_argument('--max_subgraph_size', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--estimate_only', action='store_true',
        help='Only print key/disk estimate; do not compute.',
    )
    parser.add_argument(
        '--max_keys', type=int, default=None,
        help='Cap number of keys (smoke / partial). Implies truncated table.',
    )
    parser.add_argument(
        '--refuse_above_keys', type=int, default=5_000_000,
        help='Refuse full run if cartesian keys exceed this unless --force.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Allow running even when keys > refuse_above_keys.',
    )
    parser.add_argument(
        '--bytes_per_key_est', type=int, default=10000,
        help='Disk estimate bytes per key (uci positive-only ≈ 10KB).',
    )
    parser.add_argument(
        '--out_suffix', type=str, default='',
        help='Optional suffix for output dir (e.g. smoke10k).',
    )
    main(parser.parse_args())
