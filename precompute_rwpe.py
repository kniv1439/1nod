#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_rwpe.py

⚠️ 已知问题 / 已被 precompute_rwpe_modelfn.py 取代 ⚠️
  本脚本用一份独立手写的 numpy/scipy 实现（build_neighbor_structure /
  build_center_sequence / compute_rwpe_for_sequence）来"数学上对齐"
  DyGFormerD.compute_rwpe_features，并且统一用整份 ml_{dataset}.csv（train+val+test，
  不做 new-node 排除）建图。这会造成两个问题：
    1. "两份独立实现"——查表值和在线值不是同一段代码的输出，任何实现细节出入
       都会给模型留下可学习的系统性差异（正样本查表 / 负样本在线）。
    2. "预计算图 ≠ 训练时在线图"——训练时在线负样本用的是 train_neighbor_sampler
       （仅由 train_data 构建，且 train_data 已按节点身份剔除约 10% 的
       new-test-node 及其全部边），而本脚本始终用整份 csv 建图，训练阶段会造成
       "查表(全图) vs 在线(受限图)" 的额外偏差。
  这两点会导致验证集 AP 出现不合理的接近满分（如 0.999+）。
  新代码请使用同目录下的 precompute_rwpe_modelfn.py（函数级复用 + 按 split
  选择 train_neighbor_sampler / full_neighbor_sampler），不要再用本脚本生成新表。
  本文件仅保留 build_neighbor_structure / build_center_sequence /
  compute_rwpe_for_sequence 等函数供 precompute_rwpe_node_t.py /
  precompute_rwpe_node_t_full.py 向后兼容导入，其余逻辑不建议再使用。

为 DyGFormerD 预计算 RWPE（Random Walk Positional Encoding）。

对每条交互 (src, dst, t, edge_id)：
  1. 按 pad_sequences 逻辑构造 src / dst 邻居序列（中心在前，截最近 max_seq-1）
  2. 在序列 unique 节点上建时序诱导子图（边时间 <= t）
  3. 计算 diag(RW^k), k=1..rwpe_dim，映射回序列位置
  4. 以 edge_id 为 key 写入 chunk

存储格式（chunk dict 的 value）:
  {
    'src': np.float16 array, shape (L_src, rwpe_dim),
    'dst': np.float16 array, shape (L_dst, rwpe_dim),
  }
  若两侧均无法计算（节点过少 / >max_subgraph_size / 无边），存 None。
  单侧失败时该侧存全零数组（长度与序列一致）。

Usage:
  python3 precompute_rwpe.py --dataset_name wikipedia
  python3 precompute_rwpe.py --dataset_name reddit --chunk_size 5000 --num_workers 8
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
from scipy import sparse as sp


# ---------------------------------------------------------------------------
# Worker globals
# ---------------------------------------------------------------------------

_G_node_nbr_ids = None
_G_node_nbr_times = None
_G_args = None


def _init_worker(nbr_ids, nbr_times, args):
    global _G_node_nbr_ids, _G_node_nbr_times, _G_args
    _G_node_nbr_ids = nbr_ids
    _G_node_nbr_times = nbr_times
    _G_args = args


# ---------------------------------------------------------------------------
# Neighbor structure (same as NeighborSampler / SPSE)
# ---------------------------------------------------------------------------

def build_neighbor_structure(src_ids, dst_ids, timestamps):
    max_node = int(max(src_ids.max(), dst_ids.max()))
    raw_ids = [[] for _ in range(max_node + 1)]
    raw_times = [[] for _ in range(max_node + 1)]

    for u, v, t in zip(src_ids.tolist(), dst_ids.tolist(), timestamps.tolist()):
        raw_ids[u].append(v)
        raw_times[u].append(t)
        raw_ids[v].append(u)
        raw_times[v].append(t)

    node_nbr_ids, node_nbr_times = [], []
    for u in range(max_node + 1):
        if raw_ids[u]:
            arr_ids = np.array(raw_ids[u], dtype=np.int64)
            arr_times = np.array(raw_times[u], dtype=np.float64)
            order = np.argsort(arr_times, kind='stable')
            node_nbr_ids.append(arr_ids[order])
            node_nbr_times.append(arr_times[order])
        else:
            node_nbr_ids.append(np.empty(0, dtype=np.int64))
            node_nbr_times.append(np.empty(0, dtype=np.float64))
    return node_nbr_ids, node_nbr_times


# ---------------------------------------------------------------------------
# Sequence + RWPE (aligned with DyGFormerD)
# ---------------------------------------------------------------------------

def build_center_sequence(center_id, t, node_nbr_ids, node_nbr_times, max_seq_len):
    """Return unpadded sequence: [center] + recent neighbors before t."""
    if center_id < len(node_nbr_ids):
        nbr_ids = node_nbr_ids[center_id]
        nbr_times = node_nbr_times[center_id]
        cutoff = int(np.searchsorted(nbr_times, t, side='left'))  # times < t
        nbrs = nbr_ids[:cutoff]
        if len(nbrs) > max_seq_len - 1:
            nbrs = nbrs[-(max_seq_len - 1):]
    else:
        nbrs = np.empty(0, dtype=np.int64)

    seq = np.empty(1 + len(nbrs), dtype=np.int64)
    seq[0] = center_id
    if len(nbrs) > 0:
        seq[1:] = nbrs
    return seq


def compute_rwpe_for_sequence(seq, t, node_nbr_ids, node_nbr_times, rwpe_dim, max_subgraph_size):
    """
    Compute RWPE for one padded-free sequence.
    Returns float16 array (L, rwpe_dim) or None if skipped entirely.
    """
    L = len(seq)
    pe_out = np.zeros((L, rwpe_dim), dtype=np.float32)

    valid = seq[seq != 0]
    if len(valid) < 2:
        return pe_out  # zeros, still valid length

    unique_nodes = np.unique(valid)
    num_nodes = len(unique_nodes)
    if num_nodes > max_subgraph_size:
        return pe_out  # zeros (same as DyGFormerD skip)

    node_to_local = {int(n): i for i, n in enumerate(unique_nodes)}
    unique_set = set(int(n) for n in unique_nodes)

    src_list, dst_list = [], []
    for node_id in unique_nodes:
        nid = int(node_id)
        if nid >= len(node_nbr_ids):
            continue
        neighbors = node_nbr_ids[nid]
        neighbor_times = node_nbr_times[nid]
        if len(neighbors) == 0:
            continue
        # DyGFormerD uses neighbor_times <= current_time for edges
        mask = (neighbor_times <= t)
        if not mask.any():
            continue
        cand = neighbors[mask]
        for nb in cand:
            nb = int(nb)
            if nb in unique_set:
                src_list.append(node_to_local[nid])
                dst_list.append(node_to_local[nb])

    if not src_list:
        return pe_out

    edges = set(zip(src_list, dst_list))
    src_list, dst_list = zip(*edges)
    src_arr = np.asarray(src_list, dtype=np.int64)
    dst_arr = np.asarray(dst_list, dtype=np.int64)

    try:
        A = sp.csr_matrix(
            (np.ones(len(src_arr), dtype=np.float64), (src_arr, dst_arr)),
            shape=(num_nodes, num_nodes),
        )
        in_degrees = np.asarray(A.sum(axis=0)).ravel()
        degrees = np.clip(in_degrees, 1, None)
        Dinv = sp.diags(degrees ** -1.0)
        RW = A.dot(Dinv)

        M_power = RW
        pe_cols = []
        for k in range(rwpe_dim):
            pe_cols.append(np.asarray(M_power.diagonal(), dtype=np.float32))
            if k < rwpe_dim - 1:
                M_power = M_power.dot(RW)
        PE = np.stack(pe_cols, axis=-1)  # (num_nodes, rwpe_dim)

        for seq_idx, node_id in enumerate(seq):
            nid = int(node_id)
            if nid != 0 and nid in node_to_local:
                pe_out[seq_idx] = PE[node_to_local[nid]]
    except Exception:
        return pe_out

    return pe_out.astype(np.float16)


def compute_edge_rwpe(src_id, dst_id, t, node_nbr_ids, node_nbr_times, args):
    src_seq = build_center_sequence(
        src_id, t, node_nbr_ids, node_nbr_times, args.max_input_sequence_length
    )
    dst_seq = build_center_sequence(
        dst_id, t, node_nbr_ids, node_nbr_times, args.max_input_sequence_length
    )
    src_pe = compute_rwpe_for_sequence(
        src_seq, t, node_nbr_ids, node_nbr_times, args.rwpe_dim, args.max_subgraph_size
    )
    dst_pe = compute_rwpe_for_sequence(
        dst_seq, t, node_nbr_ids, node_nbr_times, args.rwpe_dim, args.max_subgraph_size
    )
    if src_pe is None and dst_pe is None:
        return None
    return {
        'src': src_pe if src_pe is not None else np.zeros((len(src_seq), args.rwpe_dim), dtype=np.float16),
        'dst': dst_pe if dst_pe is not None else np.zeros((len(dst_seq), args.rwpe_dim), dtype=np.float16),
    }


# ---------------------------------------------------------------------------
# Chunk worker
# ---------------------------------------------------------------------------

def _process_chunk_worker(task):
    (chunk_id, chunk_start, chunk_end,
     src_ids, dst_ids, timestamps, edge_ids,
     out_dir, overwrite) = task

    chunk_path = os.path.join(out_dir, f'chunk_{chunk_id:05d}.pkl')
    if os.path.exists(chunk_path) and not overwrite:
        print(f'  Chunk {chunk_id:05d} already exists, skipping.', flush=True)
        return chunk_id, 0, 0, 0

    args = _G_args
    chunk_data = {}
    n_success = n_none = n_error = 0

    for i in range(chunk_start, chunk_end):
        eid = int(edge_ids[i])
        try:
            entry = compute_edge_rwpe(
                int(src_ids[i]), int(dst_ids[i]), float(timestamps[i]),
                _G_node_nbr_ids, _G_node_nbr_times, args,
            )
            chunk_data[eid] = entry
            if entry is None:
                n_none += 1
            else:
                n_success += 1
        except Exception as exc:
            chunk_data[eid] = None
            n_error += 1
            if getattr(args, 'verbose', False):
                print(f'    [ERROR] chunk={chunk_id} i={i} eid={eid}: {exc}', flush=True)

    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk_data, f, protocol=4)

    size = chunk_end - chunk_start
    print(
        f'  → Chunk {chunk_id:05d} done: {size} interactions  '
        f'success={n_success} none={n_none} error={n_error}',
        flush=True,
    )
    return chunk_id, n_success, n_none, n_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    t_wall = time.time()
    csv_path = os.path.join('processed_data', args.dataset_name, f'ml_{args.dataset_name}.csv')
    print(f'Loading {csv_path} ...')
    graph_df = pd.read_csv(csv_path)

    src_ids = graph_df.u.values.astype(np.int64)
    dst_ids = graph_df.i.values.astype(np.int64)
    timestamps = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.int64)
    N = len(src_ids)
    print(f'  {N} interactions, nodes up to {max(src_ids.max(), dst_ids.max())}')

    print('Building neighbor structure ...')
    node_nbr_ids, node_nbr_times = build_neighbor_structure(src_ids, dst_ids, timestamps)
    print(f'  Done. {len(node_nbr_ids)} node slots.')

    tag = f'seq{args.max_input_sequence_length}_dim{args.rwpe_dim}_ms{args.max_subgraph_size}'
    out_dir = os.path.join('processed_data', args.dataset_name, f'rwpe_precomputed_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output → {out_dir}')

    # edge_ids in interaction order — for lazy-load chunk lookup
    np.save(os.path.join(out_dir, 'edge_ids.npy'), edge_ids)

    meta = {
        'dataset_name': args.dataset_name,
        'max_input_sequence_length': args.max_input_sequence_length,
        'rwpe_dim': args.rwpe_dim,
        'max_subgraph_size': args.max_subgraph_size,
        'chunk_size': args.chunk_size,
        'num_interactions': N,
        'num_chunks': math.ceil(N / args.chunk_size),
        'time_edge_rule': 'neighbor_times <= current_time',
        'neighbor_cutoff': 'times < t (searchsorted left)',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    num_chunks = math.ceil(N / args.chunk_size)
    num_workers = min(args.num_workers, num_chunks)
    n_success = n_none = n_error = 0

    tasks = []
    for chunk_id in range(num_chunks):
        chunk_start = chunk_id * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, N)
        tasks.append((
            chunk_id, chunk_start, chunk_end,
            src_ids, dst_ids, timestamps, edge_ids,
            out_dir, args.overwrite,
        ))

    print(f'Processing {num_chunks} chunks with {num_workers} workers ...')
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(node_nbr_ids, node_nbr_times, args),
    ) as pool:
        for _, s, nn, e in pool.imap_unordered(_process_chunk_worker, tasks):
            n_success += s
            n_none += nn
            n_error += e

    elapsed = time.time() - t_wall
    print(f'\nFinished. total={N} success={n_success} none={n_none} error={n_error}')
    print(f'Total time: {elapsed / 60:.1f} min  |  Output: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute RWPE for DyGFormerD.')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--max_input_sequence_length', type=int, default=512)
    parser.add_argument('--rwpe_dim', type=int, default=8)
    parser.add_argument('--max_subgraph_size', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
