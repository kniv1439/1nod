#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_spd_modelfn.py

为 DyGFormerST 预计算 SPD（最短路径距离编码）——函数级复用 + 按 split 对齐图（edge_id 键）。

设计原则跟 precompute_rwpe_modelfn.py 完全一致：
  1. 直接复用 DyGFormerST 自身的 pad_sequences + compute_spatial_pos_dist
     （不是另写一份独立实现），离线表和在线 miss 回退 / 负样本在线计算走的是
     完全同一份代码，消除"两份独立实现"的潜在偏差；
  2. train split 的边用 train_neighbor_sampler 建图，val/test split 用
     full_neighbor_sampler 建图，与 train_link_prediction.py /
     evaluate_link_prediction.py 在线时使用的图完全一致，避免图源泄漏；
  3. key_type = edge_id：只覆盖正样本（数据集里真实发生的边）。负样本（假边）
     没有真实 edge_id，训练/评估时会传 edge_ids=None，整体走
     DyGFormerST.compute_spatial_pos_dist 在线计算（src、dst 两侧都不查表），
     跟 DyGFormerD 旧版 edge_id 模式的 RWPE 完全同构。

存储格式：
  chunk_{id:05d}.pkl: {edge_id(int): {'src': np.int16[Ls], 'dst': np.int16[Ld]}}
      其中 -1 表示该位置"无信号"（padding / 有效节点数<2 / 子图超过 max_subgraph_size），
      0..511 表示真实最短路径距离。
  meta.json: key_type='edge_id' 及各项一致性参数。
  edge_ids.npy: 按 train→val→test 拼接后的全局行顺序排列的 edge_id
  （供 lazy-load 模式按 edge_id 定位 chunk；行号到 chunk 的映射为
  row // chunk_size，因此一个 chunk 内部可能同时包含来自两个 split 的行，
  已在计算阶段按行归属正确地分别使用对应的 neighbor_sampler）。

多进程说明：与 precompute_rwpe_modelfn.py 完全一致——Linux 下 fork 启动方式，
父进程构建好的 NeighborSampler / 壳模型通过写时复制被子进程继承；全程 CPU 计算
（compute_spatial_pos_dist 里的 Floyd-Warshall 本来就是 CPU/numpy 运算）。

Usage:
  python3 precompute_spd_modelfn.py --dataset_name uci --estimate_only
  python3 precompute_spd_modelfn.py --dataset_name uci --num_workers 8
  python3 precompute_spd_modelfn.py --dataset_name uci --chunk_size 2000 --num_workers 16 --overwrite
"""

import os
import json
import time
import math
import argparse
import pickle
import multiprocessing as mp

import numpy as np
import torch

from utils.DataLoader import get_link_prediction_data
from utils.utils import get_neighbor_sampler
from models.DyGFormerST import DyGFormerST


# ---------------------------------------------------------------------------
# Worker globals (populated once per worker via Pool initializer; on Linux
# 'fork' these are inherited via copy-on-write from the parent process, no
# pickling of the (potentially large) model / arrays is required)
# ---------------------------------------------------------------------------

_G_model = None
_G_train_sampler = None
_G_full_sampler = None
_G_n_train = None
_G_src_ids = None
_G_dst_ids = None
_G_times = None
_G_edge_ids = None


def _init_worker(model: DyGFormerST, train_sampler, full_sampler, n_train: int,
                  src_ids: np.ndarray, dst_ids: np.ndarray,
                  times: np.ndarray, edge_ids: np.ndarray):
    global _G_model, _G_train_sampler, _G_full_sampler, _G_n_train
    global _G_src_ids, _G_dst_ids, _G_times, _G_edge_ids
    _G_model = model
    _G_train_sampler = train_sampler
    _G_full_sampler = full_sampler
    _G_n_train = n_train
    _G_src_ids = src_ids
    _G_dst_ids = dst_ids
    _G_times = times
    _G_edge_ids = edge_ids
    # 保险起见强制单线程 BLAS/OMP，避免多进程 * 多线程矩阵库互相抢核
    torch.set_num_threads(1)


def _compute_side_spd(model: DyGFormerST, node_ids: np.ndarray, times: np.ndarray):
    """
    对一批 (node_id, t) 调用与训练时完全相同的两个模型方法：
      get_all_first_hop_neighbors -> pad_sequences -> compute_spatial_pos_dist
    这里的 model.neighbor_sampler 由调用方（_accumulate_range）按行所属 split
    提前切换为 train_neighbor_sampler 或 full_neighbor_sampler。
    :return: padded_ids ndarray (B, seq_len), dist ndarray[int16] (B, seq_len)
    """
    nbr_ids_list, edge_ids_list, nbr_times_list = \
        model.neighbor_sampler.get_all_first_hop_neighbors(node_ids=node_ids, node_interact_times=times)

    padded_ids, _, _ = model.pad_sequences(
        node_ids=node_ids, node_interact_times=times,
        nodes_neighbor_ids_list=nbr_ids_list, nodes_edge_ids_list=edge_ids_list,
        nodes_neighbor_times_list=nbr_times_list,
        patch_size=model.patch_size, max_input_sequence_length=model.max_input_sequence_length,
    )

    dist = model.compute_spatial_pos_dist(
        padded_nodes_neighbor_ids=padded_ids,
        node_ids=node_ids,
        node_interact_times=times,
    )
    return padded_ids, dist


def _accumulate_range(chunk_data: dict, sampler, start: int, end: int):
    """对全局行区间 [start, end) 用指定 sampler 计算 SPD 并写入 chunk_data。"""
    if start >= end:
        return
    model = _G_model
    model.set_neighbor_sampler(sampler)

    src_batch = _G_src_ids[start:end]
    dst_batch = _G_dst_ids[start:end]
    t_batch = _G_times[start:end]
    eid_batch = _G_edge_ids[start:end]

    with torch.no_grad():
        src_padded_ids, src_dist = _compute_side_spd(model, src_batch, t_batch)
        dst_padded_ids, dst_dist = _compute_side_spd(model, dst_batch, t_batch)

    for b in range(len(eid_batch)):
        ls = model._valid_seq_len(src_padded_ids[b])
        ld = model._valid_seq_len(dst_padded_ids[b])
        src_arr = src_dist[b, :ls].astype(np.int16) if ls > 0 else np.empty(0, dtype=np.int16)
        dst_arr = dst_dist[b, :ld].astype(np.int16) if ld > 0 else np.empty(0, dtype=np.int16)
        chunk_data[int(eid_batch[b])] = {'src': src_arr, 'dst': dst_arr}


def _process_chunk(task):
    """
    task = (chunk_id, lo, hi)，[lo, hi) 是拼接后 (train→val→test) 的全局行区间。
    一个 chunk 内部可能跨越 train / val|test 边界，因此按 _G_n_train 拆成
    最多两段，分别用 train_neighbor_sampler / full_neighbor_sampler 计算，
    再合并写入同一个 chunk 文件——保证每一行都用了"和它所属 split 在线时
    完全一致"的图。
    """
    chunk_id, lo, hi = task
    chunk_data = {}

    cut = min(max(_G_n_train, lo), hi)
    # [lo, cut) 落在 train split 内 -> train_neighbor_sampler
    _accumulate_range(chunk_data, _G_train_sampler, lo, cut)
    # [cut, hi) 落在 val/test split 内 -> full_neighbor_sampler
    _accumulate_range(chunk_data, _G_full_sampler, cut, hi)

    return chunk_id, chunk_data, hi - lo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_shell_model(args, node_raw_features, edge_raw_features, neighbor_sampler):
    """构建一个只用于离线计算 SPD 的 DyGFormerST 实例（不训练、不加载权重）。"""
    model = DyGFormerST(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=args.time_feat_dim,
        channel_embedding_dim=args.channel_embedding_dim,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_input_sequence_length=args.max_input_sequence_length,
        device='cpu',
        max_subgraph_size=args.max_subgraph_size,
        spd_precomputed_dir=None,
        spd_lazy_load=False,
    )
    model.eval()
    return model


def main(args):
    t_wall = time.time()

    print(f'Loading link prediction data for {args.dataset_name} ...')
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, \
        new_node_val_data, new_node_test_data = get_link_prediction_data(
            dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        )

    n_train, n_val, n_test = len(train_data.src_node_ids), len(val_data.src_node_ids), len(test_data.src_node_ids)
    N = n_train + n_val + n_test
    print(f'  train={n_train}  val={n_val}  test={n_test}  (total to precompute: {N})')
    print(f'  (new_node_val={len(new_node_val_data.src_node_ids)} / '
          f'new_node_test={len(new_node_test_data.src_node_ids)} 是 val/test 按 edge_id 的子集，自动覆盖)')

    print('Building NeighborSampler (utils.utils.get_neighbor_sampler, 与训练脚本完全同一份构造方式) ...')
    # 与 train_link_prediction.py 中 train_neighbor_sampler / full_neighbor_sampler 的构造完全一致
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    print('Building shell DyGFormerST model (CPU, eval-only) ...')
    model = build_shell_model(args, node_raw_features, edge_raw_features, train_neighbor_sampler)

    tag = f'seq{args.max_input_sequence_length}_ps{args.patch_size}_ms{args.max_subgraph_size}'
    out_dir = os.path.join('processed_data', args.dataset_name, f'spd_precomputed_modelfn_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output → {out_dir}')

    # 全局行顺序：train → val → test，与 train_link_prediction.py /
    # evaluate_link_prediction.py 里各自读取 train_data/val_data/test_data.edge_ids
    # 的 split 划分完全对应。edge_ids.npy 按这个顺序保存，供 lazy-load 的
    # row -> chunk_id (= row // chunk_size) 查找使用。
    src_ids = np.concatenate([train_data.src_node_ids, val_data.src_node_ids, test_data.src_node_ids]).astype(np.int64)
    dst_ids = np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids, test_data.dst_node_ids]).astype(np.int64)
    times = np.concatenate([train_data.node_interact_times, val_data.node_interact_times, test_data.node_interact_times]).astype(np.float64)
    edge_ids = np.concatenate([train_data.edge_ids, val_data.edge_ids, test_data.edge_ids]).astype(np.int64)
    assert len(src_ids) == N

    np.save(os.path.join(out_dir, 'edge_ids.npy'), edge_ids)

    num_chunks = math.ceil(N / args.chunk_size)
    meta = {
        'dataset_name': args.dataset_name,
        'key_type': 'edge_id',
        'max_input_sequence_length': args.max_input_sequence_length,
        'patch_size': args.patch_size,
        'max_subgraph_size': args.max_subgraph_size,
        'chunk_size': args.chunk_size,
        'num_interactions': N,
        'num_train': n_train,
        'num_val': n_val,
        'num_test': n_test,
        'num_chunks': num_chunks,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'sample_neighbor_strategy': args.sample_neighbor_strategy,
        'time_scaling_factor': args.time_scaling_factor,
        'generator': 'precompute_spd_modelfn.py (split-aware: reuses DyGFormerST.pad_sequences + '
                     'compute_spatial_pos_dist + get_neighbor_sampler; train edges computed with '
                     'train_neighbor_sampler, val/test edges computed with full_neighbor_sampler, '
                     'matching exactly what train_link_prediction.py / evaluate_link_prediction.py '
                     'use online for the same split)',
        'graph_source': 'split-aware (train_neighbor_sampler for train edges; full_neighbor_sampler for val/test edges)',
        'quantization': 'int16 raw shortest-path distance, -1 = no signal (padding/degenerate/oversized subgraph)',
        'negative_samples': 'not covered (edge_id key); at train/eval time pass edge_ids=None for negatives '
                             'so both src and dst sides fall back fully online via the same compute_spatial_pos_dist',
        'row_order': 'train_data rows, then val_data rows, then test_data rows (matches edge_ids.npy)',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    if args.estimate_only:
        print('\n[estimate_only] Timing a single chunk to estimate total runtime ...')
        _init_worker(model, train_neighbor_sampler, full_neighbor_sampler, n_train,
                     src_ids, dst_ids, times, edge_ids)
        probe_hi = min(args.chunk_size, N)
        t0 = time.time()
        _process_chunk((0, 0, probe_hi))
        dt = time.time() - t0
        per_edge = dt / max(probe_hi, 1)
        est_total = per_edge * N
        print(f'  1 chunk ({probe_hi} edges) took {dt:.1f}s -> {per_edge * 1000:.2f} ms/edge')
        print(f'  Estimated single-process total: {est_total / 60:.1f} min '
              f'(~{est_total / 60 / max(args.num_workers, 1):.1f} min with {args.num_workers} workers, ideal scaling)')
        return

    tasks = []
    for chunk_id in range(num_chunks):
        lo = chunk_id * args.chunk_size
        hi = min(lo + args.chunk_size, N)
        chunk_path = os.path.join(out_dir, f'chunk_{chunk_id:05d}.pkl')
        if os.path.exists(chunk_path) and not args.overwrite:
            continue
        tasks.append((chunk_id, lo, hi))

    if not tasks:
        print('All chunks already exist. Use --overwrite to recompute.')
        return

    num_workers = max(1, min(args.num_workers, len(tasks)))
    print(f'Processing {len(tasks)}/{num_chunks} chunks with {num_workers} worker(s) (device=cpu) ...')

    ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else mp.get_context()
    n_done_edges = 0
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(model, train_neighbor_sampler, full_neighbor_sampler, n_train,
                  src_ids, dst_ids, times, edge_ids),
    ) as pool:
        for chunk_id, chunk_data, n_edges in pool.imap_unordered(_process_chunk, tasks):
            chunk_path = os.path.join(out_dir, f'chunk_{chunk_id:05d}.pkl')
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_data, f, protocol=4)
            n_done_edges += n_edges
            print(f'  -> Chunk {chunk_id:05d} done ({n_edges} interactions, '
                  f'{n_done_edges}/{N} total so far)', flush=True)

    elapsed = time.time() - t_wall
    print(f'\nFinished. total_interactions={N} (train={n_train}, val={n_val}, test={n_test}) '
          f'chunks_written={len(tasks)}')
    print(f'Total time: {elapsed / 60:.1f} min  |  Output: {out_dir}')
    print('Train with: --spd_precomputed_dir', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute SPD for DyGFormerST by directly reusing '
                     'DyGFormerST.compute_spatial_pos_dist + get_neighbor_sampler (function-level reuse), '
                     'with per-split graph selection (train_neighbor_sampler for train edges, '
                     'full_neighbor_sampler for val/test edges) to avoid new-node graph leakage. '
                     'edge_id key: only positive samples are covered; negative samples must pass '
                     'edge_ids=None at train/eval time and compute fully online.'
    )
    parser.add_argument('--dataset_name', type=str, required=True)

    # SPD / 序列参数：需与训练命令行的对应参数保持一致
    parser.add_argument('--max_input_sequence_length', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--max_subgraph_size', type=int, default=512)

    # DyGFormerST 构造所需的其余参数（不影响 SPD 数值，仅用于成功实例化壳模型）
    parser.add_argument('--time_feat_dim', type=int, default=100)
    parser.add_argument('--channel_embedding_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    # 邻居采样策略：需与训练命令行一致
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent',
                         choices=['uniform', 'recent', 'time_interval_aware'])
    parser.add_argument('--time_scaling_factor', type=float, default=1e-6)

    # 数据集切分比例：需与训练/评估脚本一致，因为现在会按 split 分别建图
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)

    parser.add_argument('--chunk_size', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--estimate_only', action='store_true',
                         help='只算 1 个 chunk 估算总耗时，不生成完整预计算结果')

    main(parser.parse_args())
