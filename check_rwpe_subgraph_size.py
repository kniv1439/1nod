# -*- coding: utf-8 -*-
"""
检查 DyGFormerD 的 RWPE 子图 unique 节点数是否会超过 MAX_SUBGRAPH_SIZE=200。

对齐 models/DyGFormerD.py:
  - 序列 = [中心节点] + 最近 (max_input_sequence_length-1) 个一跳历史邻居
  - 子图节点数 = len(unique(序列))
  - src / dst 分开算（不是并集）

仅依赖 numpy / pandas，不依赖 torch。
"""
import os
import numpy as np
import pandas as pd

DATASETS = [
    'wikipedia', 'reddit', 'uci', 'enron', 'mooc',
    'USLegis', 'UNtrade', 'UNvote', 'CanParl',
]
MAX_INPUT_SEQUENCE_LENGTH = 512  # DyGFormerD 默认
MAX_SUBGRAPH_SIZE = 200
MAX_NEIGHBORS = MAX_INPUT_SEQUENCE_LENGTH - 1


def build_adj_lists(src_ids, dst_ids, times):
    """无向邻接：每个节点存 (neighbor_id, time)，并按时间排序。"""
    max_node_id = int(max(src_ids.max(), dst_ids.max()))
    adj = [[] for _ in range(max_node_id + 1)]
    for s, d, t in zip(src_ids, dst_ids, times):
        s, d, t = int(s), int(d), float(t)
        adj[s].append((d, t))
        adj[d].append((s, t))
    # 按时间排序，便于 searchsorted
    nei_ids = [None] * (max_node_id + 1)
    nei_times = [None] * (max_node_id + 1)
    for nid in range(max_node_id + 1):
        if not adj[nid]:
            nei_ids[nid] = np.array([], dtype=np.int64)
            nei_times[nid] = np.array([], dtype=np.float64)
            continue
        arr = sorted(adj[nid], key=lambda x: x[1])
        nei_ids[nid] = np.array([x[0] for x in arr], dtype=np.int64)
        nei_times[nid] = np.array([x[1] for x in arr], dtype=np.float64)
    return nei_ids, nei_times


def neighbors_before(nei_ids, nei_times, node_id, interact_time):
    """返回 interact_time 之前的全部一跳邻居（已按时间排序）。"""
    times = nei_times[node_id]
    if len(times) == 0:
        return np.array([], dtype=np.int64)
    # 严格与 NeighborSampler.find_neighbors_before 一致：searchsorted 默认 left，
    # 即 times < interact_time（不含等于）。DyGFormer 取邻居时也是 before。
    # 但 RWPE 建边用的是 <= current_time。节点集合用 before 邻居序列即可。
    i = int(np.searchsorted(times, interact_time))
    return nei_ids[node_id][:i]


def unique_subgraph_size(neighbor_ids, center_id):
    """与 DyGFormerD.pad_sequences + compute_rwpe_features 一致。"""
    if neighbor_ids is None or len(neighbor_ids) == 0:
        return 1
    if len(neighbor_ids) > MAX_NEIGHBORS:
        neighbor_ids = neighbor_ids[-MAX_NEIGHBORS:]
    seq = np.concatenate([[center_id], neighbor_ids])
    seq = seq[seq != 0]
    return int(len(np.unique(seq)))


def check_dataset(dataset_name):
    print(f'\n===== {dataset_name} =====', flush=True)
    csv_path = f'./processed_data/{dataset_name}/ml_{dataset_name}.csv'
    graph_df = pd.read_csv(csv_path)
    src_ids = graph_df.u.values.astype(np.int64)
    dst_ids = graph_df.i.values.astype(np.int64)
    times = graph_df.ts.values.astype(np.float64)
    n = len(src_ids)
    print(f'  interactions={n}, building adj lists...', flush=True)

    nei_ids, nei_times = build_adj_lists(src_ids, dst_ids, times)

    src_sizes = np.empty(n, dtype=np.int32)
    dst_sizes = np.empty(n, dtype=np.int32)
    report_every = max(n // 20, 1)

    for i in range(n):
        if i % report_every == 0:
            print(f'  [{dataset_name}] {i}/{n} ({100.0 * i / n:.1f}%)', flush=True)

        s = int(src_ids[i])
        d = int(dst_ids[i])
        t = float(times[i])

        s_nei = neighbors_before(nei_ids, nei_times, s, t)
        d_nei = neighbors_before(nei_ids, nei_times, d, t)

        src_sizes[i] = unique_subgraph_size(s_nei, s)
        dst_sizes[i] = unique_subgraph_size(d_nei, d)

    all_sizes = np.concatenate([src_sizes, dst_sizes])
    over_src = int((src_sizes > MAX_SUBGRAPH_SIZE).sum())
    over_dst = int((dst_sizes > MAX_SUBGRAPH_SIZE).sum())
    over_either = int(((src_sizes > MAX_SUBGRAPH_SIZE) | (dst_sizes > MAX_SUBGRAPH_SIZE)).sum())

    summary = {
        'dataset': dataset_name,
        'num_edges': n,
        'max_unique': int(all_sizes.max()),
        'p50': float(np.median(all_sizes)),
        'p90': float(np.percentile(all_sizes, 90)),
        'p99': float(np.percentile(all_sizes, 99)),
        'mean': float(all_sizes.mean()),
        'src_over_200': over_src,
        'dst_over_200': over_dst,
        'edge_either_over_200': over_either,
        'edge_either_over_200_ratio': over_either / n,
        'center_seqs_over_200': over_src + over_dst,
        'center_seqs_over_200_ratio': (over_src + over_dst) / (2 * n),
    }

    print(
        f"edges={n} | max_unique={summary['max_unique']} | "
        f"p50={summary['p50']:.1f} p90={summary['p90']:.1f} p99={summary['p99']:.1f} | "
        f"src>200={over_src} dst>200={over_dst} | "
        f"edge(src|dst)>200={over_either} ({100 * summary['edge_either_over_200_ratio']:.2f}%)",
        flush=True,
    )
    return summary


if __name__ == '__main__':
    rows = []
    for name in DATASETS:
        path = f'./processed_data/{name}/ml_{name}.csv'
        if not os.path.exists(path):
            print(f'[SKIP] missing {path}')
            continue
        rows.append(check_dataset(name))

    print('\n===== Summary =====')
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out_csv = 'rwpe_subgraph_size_check.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved to {out_csv}')
