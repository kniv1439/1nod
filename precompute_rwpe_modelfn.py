#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_rwpe_modelfn.py

为 DyGFormerD 预计算 RWPE（Random Walk Positional Encoding）——"函数级复用 + 按 split 对齐图"版本。

与 precompute_rwpe.py 的区别（解决"路径 A：两份独立实现"）：
  - precompute_rwpe.py 是另一份独立手写的实现（自建邻居结构 + 手写 RW^k 逻辑），
    只是力求在数学上与 DyGFormerD.compute_rwpe_features 等价，但两份代码分别维护，
    任何实现细节出入都会给模型留下"查表值"和"在线值"之间可被学习的系统性差异。
  - 本脚本直接：
      1. 用 utils.utils.get_neighbor_sampler 构建 NeighborSampler
         （与 train_link_prediction.py 里构造 train_neighbor_sampler / full_neighbor_sampler
         完全同一份代码、同一构造方式）；
      2. 实例化一个不参与训练、只用于计算的"壳" DyGFormerD；
      3. 对每条边调用该实例的 pad_sequences + compute_rwpe_features
         ——与训练/评估时负样本在线计算调用的是同一个函数、同一份代码。
    从而消除"两份独立实现"的潜在偏差，正样本(缓存值)和负样本(在线值)在算法层面
    严格是同一份代码的输出。

与旧版 precompute_rwpe_modelfn.py 的区别（解决"路径 B：预计算图 ≠ 训练时在线图"，
new-node 泄漏，训练阶段独有）：
  - 旧版对所有 split（train/val/test）统一用 full_neighbor_sampler（基于 full_data，
    即整份 csv，不做任何 new-node 排除）计算 RWPE。但训练时在线负样本走的是
    train_neighbor_sampler（仅基于 train_data 构建，且 train_data 在划分阶段已经
    按"节点身份"剔除了约 10% 的 new-test-node 及其全部边，参见
    utils/DataLoader.py::get_link_prediction_data 的 observed_edges_mask / train_mask）。
    即便算法实现完全对齐，只要图源不同，同一个 (src, t) 在"查表"和"在线"两条路径
    下也可能得到不同的子图结构——这在训练阶段会给模型留下一个与真实链路结构无关、
    只反映"这条边走的是查表还是在线"的捷径特征。
  - 本版本按 split 使用与在线完全一致的图：
      * train split 的边 → train_neighbor_sampler（与训练循环中
        set_neighbor_sampler(train_neighbor_sampler) 之后在线计算负样本时用的图完全一致）；
      * val / test split 的边（new_node_val / new_node_test 是它们按 edge_id 的子集，
        自动覆盖，无需单独处理）→ full_neighbor_sampler（与评估时
        neighbor_sampler=full_neighbor_sampler 完全一致）。
    这样"查表命中"在数值上等价于"假如在线算一遍"，命中/未命中只是缓存加速，
    不再是可被模型利用的泄漏源。

存储格式与 precompute_rwpe.py 保持一致（对 _load_rwpe_from_precomp 完全兼容）：
  chunk_{id:05d}.pkl: {edge_id(int): {'src': np.float16 (L_src, rwpe_dim),
                                       'dst': np.float16 (L_dst, rwpe_dim)}}
  meta.json: 记录 key_type='edge_id' 及各项一致性参数。
  edge_ids.npy: 按 train→val→test 拼接后的全局行顺序排列的 edge_id
  （供 lazy-load 模式按 edge_id 定位 chunk；行号到 chunk 的映射为
  row // chunk_size，因此一个 chunk 内部可能同时包含来自两个 split 的行，
  已在计算阶段按行归属正确地分别使用对应的 neighbor_sampler）。

多进程说明：
  - Linux 下默认 'fork' 启动方式，父进程构建好的 NeighborSampler / 壳模型 通过写时复制
    直接被子进程继承，不需要显式 pickle 大对象，开销很小。
  - 计算全程使用 CPU（device='cpu'）：compute_rwpe_features 里真正耗时的
    scipy 稀疏矩阵运算本来就在 CPU 上执行，GPU 对此没有加速作用，
    多进程用 CPU 还能避免多个 worker 抢占同一张 GPU / 显存的问题。
  - 每个 worker 共享同一个壳模型对象，通过 model.set_neighbor_sampler(...) 在
    train_neighbor_sampler 与 full_neighbor_sampler 之间按行区间切换，
    两者互不干扰（同一 worker 内部严格串行处理，不存在并发覆盖问题）。

Usage:
  python3 precompute_rwpe_modelfn.py --dataset_name uci --estimate_only
  python3 precompute_rwpe_modelfn.py --dataset_name uci --num_workers 8
  python3 precompute_rwpe_modelfn.py --dataset_name uci --chunk_size 2000 --num_workers 16 --overwrite
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
from models.DyGFormerD import DyGFormerD


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


def _init_worker(model: DyGFormerD, train_sampler, full_sampler, n_train: int,
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
    # 保险起见强制单线程 BLAS/OMP，避免多进程 * 多线程矩阵库 互相抢核
    torch.set_num_threads(1)


def _compute_side_rwpe(model: DyGFormerD, node_ids: np.ndarray, times: np.ndarray):
    """
    对一批 (node_id, t) 调用与训练时完全相同的两个模型方法：
      get_all_first_hop_neighbors -> pad_sequences -> compute_rwpe_features
    这里的 model.neighbor_sampler 由调用方（_accumulate_range）按行所属 split
    提前切换为 train_neighbor_sampler 或 full_neighbor_sampler。
    :return: padded_ids ndarray (B, seq_len), pe Tensor (B, seq_len, rwpe_dim)
    """
    nbr_ids_list, edge_ids_list, nbr_times_list = \
        model.neighbor_sampler.get_all_first_hop_neighbors(node_ids=node_ids, node_interact_times=times)

    padded_ids, _, _ = model.pad_sequences(
        node_ids=node_ids, node_interact_times=times,
        nodes_neighbor_ids_list=nbr_ids_list, nodes_edge_ids_list=edge_ids_list,
        nodes_neighbor_times_list=nbr_times_list,
        patch_size=model.patch_size, max_input_sequence_length=model.max_input_sequence_length,
    )

    pe = model.compute_rwpe_features(
        padded_nodes_neighbor_ids=padded_ids,
        nodes_neighbor_ids_list=nbr_ids_list,
        node_ids=node_ids,
        node_interact_times=times,
    )
    return padded_ids, pe


def _accumulate_range(chunk_data: dict, sampler, start: int, end: int):
    """对全局行区间 [start, end) 用指定 sampler 计算 RWPE 并写入 chunk_data。"""
    if start >= end:
        return
    model = _G_model
    model.set_neighbor_sampler(sampler)

    src_batch = _G_src_ids[start:end]
    dst_batch = _G_dst_ids[start:end]
    t_batch = _G_times[start:end]
    eid_batch = _G_edge_ids[start:end]

    with torch.no_grad():
        src_padded_ids, src_pe = _compute_side_rwpe(model, src_batch, t_batch)
        dst_padded_ids, dst_pe = _compute_side_rwpe(model, dst_batch, t_batch)

    d = model.rwpe_dim
    for b in range(len(eid_batch)):
        ls = model._valid_seq_len(src_padded_ids[b])
        ld = model._valid_seq_len(dst_padded_ids[b])
        src_arr = src_pe[b, :ls].cpu().numpy().astype(np.float16) if ls > 0 else np.zeros((0, d), dtype=np.float16)
        dst_arr = dst_pe[b, :ld].cpu().numpy().astype(np.float16) if ld > 0 else np.zeros((0, d), dtype=np.float16)
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
    """构建一个只用于离线计算 RWPE 的 DyGFormerD 实例（不训练、不加载权重）。"""
    model = DyGFormerD(
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
        use_rpe=True,
        rwpe_dim=args.rwpe_dim,
        max_subgraph_size=args.max_subgraph_size,
        rwpe_precomputed_dir=None,
        rwpe_lazy_load=False,
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

    print('Building shell DyGFormerD model (CPU, eval-only) ...')
    model = build_shell_model(args, node_raw_features, edge_raw_features, train_neighbor_sampler)

    tag = f'seq{args.max_input_sequence_length}_dim{args.rwpe_dim}_ms{args.max_subgraph_size}'
    out_dir = os.path.join('processed_data', args.dataset_name, f'rwpe_precomputed_modelfn_{tag}')
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
        'rwpe_dim': args.rwpe_dim,
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
        'generator': 'precompute_rwpe_modelfn.py (split-aware: reuses DyGFormerD.pad_sequences + '
                     'compute_rwpe_features + get_neighbor_sampler; train edges computed with '
                     'train_neighbor_sampler, val/test edges computed with full_neighbor_sampler, '
                     'matching exactly what train_link_prediction.py / evaluate_link_prediction.py '
                     'use online for the same split)',
        'graph_source': 'split-aware (train_neighbor_sampler for train edges; full_neighbor_sampler for val/test edges)',
        'quantization': 'float16 (matches online compute_rwpe_features().half().float() path)',
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
    print('Train with: --rwpe_precomputed_dir', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute RWPE for DyGFormerD by directly reusing '
                     'DyGFormerD.compute_rwpe_features + get_neighbor_sampler (function-level reuse), '
                     'with per-split graph selection (train_neighbor_sampler for train edges, '
                     'full_neighbor_sampler for val/test edges) to avoid new-node graph leakage.'
    )
    parser.add_argument('--dataset_name', type=str, required=True)

    # RWPE / 序列参数：需与训练命令行的对应参数保持一致
    parser.add_argument('--max_input_sequence_length', type=int, default=512)
    parser.add_argument('--rwpe_dim', type=int, default=8)
    parser.add_argument('--max_subgraph_size', type=int, default=512)

    # DyGFormerD 构造所需的其余参数（不影响 RWPE 数值，仅用于成功实例化壳模型）
    # 默认值与 utils/load_configs.py 中 DyGFormerD 组的 load_best_configs 默认一致
    parser.add_argument('--time_feat_dim', type=int, default=100)
    parser.add_argument('--channel_embedding_dim', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    # 邻居采样策略：需与训练命令行一致（默认 'recent'，与 DyGFormerD 默认配置一致）
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
