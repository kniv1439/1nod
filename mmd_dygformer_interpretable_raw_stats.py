# -*- coding: utf-8 -*-
"""
Compute train/test MMD distance with interpretable DyGFormer-style raw history stats.

This script does not use trained DyGFormer parameters. For each interaction (src, dst, t),
it builds features from source/destination historical neighbors before t:
  - node: raw node features of self node + historical neighbor nodes
  - edge: raw edge features of historical interaction edges
  - time: summary statistics of historical time deltas
  - co: raw co-occurrence statistics between src/dst historical neighbor sets
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from mmd_dataset_comparison_mmd2 import DATASETS, normalize_by_train
from utils.DataLoader import Data
from utils.utils import get_neighbor_sampler


NODE_FEAT_DIM = 172
EDGE_FEAT_DIM = 172
TARGET_DATASETS = {
    "CanParl",
    "enron",
    "mooc",
    "reddit",
    "uci",
    "UNtrade",
    "UNvote",
    "USLegis",
    "wikipedia",
}


def pad_feature_dim(features: np.ndarray, target_dim: int, name: str) -> np.ndarray:
    if features.shape[1] > target_dim:
        raise ValueError(f"{name} feature dim {features.shape[1]} is larger than target dim {target_dim}")
    if features.shape[1] == target_dim:
        return features.astype(np.float32)
    padding = np.zeros((features.shape[0], target_dim - features.shape[1]), dtype=features.dtype)
    return np.concatenate([features, padding], axis=1).astype(np.float32)


def load_raw_features(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    node_feat = np.load(f"processed_data/{dataset_name}/ml_{dataset_name}_node.npy")
    edge_feat = np.load(f"processed_data/{dataset_name}/ml_{dataset_name}.npy")
    return (
        pad_feature_dim(node_feat, NODE_FEAT_DIM, f"{dataset_name} node"),
        pad_feature_dim(edge_feat, EDGE_FEAT_DIM, f"{dataset_name} edge"),
    )


def build_data_from_df(df: pd.DataFrame) -> Data:
    labels = df["label"].to_numpy() if "label" in df.columns else np.zeros(len(df), dtype=np.float32)
    return Data(
        src_node_ids=df["u"].to_numpy(dtype=np.longlong),
        dst_node_ids=df["i"].to_numpy(dtype=np.longlong),
        node_interact_times=df["ts"].to_numpy(dtype=np.float64),
        edge_ids=df["idx"].to_numpy(dtype=np.longlong),
        labels=labels,
    )


def summarize_matrix(mat: np.ndarray, dim: int) -> np.ndarray:
    """Return mean/std/min/max for a history matrix."""
    if len(mat) == 0:
        return np.zeros(4 * dim, dtype=np.float32)
    return np.concatenate(
        [
            mat.mean(axis=0),
            mat.std(axis=0),
            mat.min(axis=0),
            mat.max(axis=0),
        ]
    ).astype(np.float32)


def summarize_time_deltas(deltas: np.ndarray) -> np.ndarray:
    """Return direct and log1p time-delta summary stats."""
    if len(deltas) == 0:
        return np.zeros(8, dtype=np.float32)

    x = np.asarray(deltas, dtype=np.float32)
    x = np.maximum(x, 0.0)
    lx = np.log1p(x)
    return np.array(
        [
            x.mean(),
            x.std(),
            x.min(),
            x.max(),
            lx.mean(),
            lx.std(),
            lx.min(),
            lx.max(),
        ],
        dtype=np.float32,
    )


def _stats(values) -> list:
    if len(values) == 0:
        return [0.0, 0.0, 0.0]
    arr = np.asarray(values, dtype=np.float32)
    return [float(arr.mean()), float(arr.std()), float(arr.max())]


def cooccurrence_stats(src_hist_nodes: np.ndarray, dst_hist_nodes: np.ndarray) -> np.ndarray:
    """
    Raw co-occurrence stats between src/dst historical neighbor sequences.

    Output:
      src_self_count mean/std/max,
      dst_self_count mean/std/max,
      src_in_dst count mean/std/max,
      dst_in_src count mean/std/max,
      common unique count,
      jaccard over unique historical neighbors.
    """
    src_hist_nodes = np.asarray(src_hist_nodes, dtype=np.longlong)
    dst_hist_nodes = np.asarray(dst_hist_nodes, dtype=np.longlong)

    src_counter = Counter(src_hist_nodes.tolist())
    dst_counter = Counter(dst_hist_nodes.tolist())

    src_self_counts = [src_counter[n] for n in src_hist_nodes]
    dst_self_counts = [dst_counter[n] for n in dst_hist_nodes]
    src_in_dst = [dst_counter.get(n, 0) for n in src_hist_nodes]
    dst_in_src = [src_counter.get(n, 0) for n in dst_hist_nodes]

    src_set = set(src_counter.keys())
    dst_set = set(dst_counter.keys())
    union_size = len(src_set | dst_set)
    common_size = len(src_set & dst_set)
    jaccard = common_size / union_size if union_size > 0 else 0.0

    return np.array(
        _stats(src_self_counts)
        + _stats(dst_self_counts)
        + _stats(src_in_dst)
        + _stats(dst_in_src)
        + [float(common_size), float(jaccard)],
        dtype=np.float32,
    )


def maybe_recent_limit(
    nodes: np.ndarray,
    edges: np.ndarray,
    times: np.ndarray,
    max_history: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_history is None or max_history <= 0 or len(nodes) <= max_history:
        return nodes, edges, times
    order = np.argsort(times)[-max_history:]
    return nodes[order], edges[order], times[order]


def sample_data(data: Data, max_interactions: Optional[int], seed: int) -> Data:
    if max_interactions is None or max_interactions <= 0 or data.num_interactions <= max_interactions:
        return data

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(data.num_interactions, size=max_interactions, replace=False))
    return Data(
        src_node_ids=data.src_node_ids[idx],
        dst_node_ids=data.dst_node_ids[idx],
        node_interact_times=data.node_interact_times[idx],
        edge_ids=data.edge_ids[idx],
        labels=data.labels[idx],
    )


def extract_interpretable_features(
    data: Data,
    sampler,
    node_raw: np.ndarray,
    edge_raw: np.ndarray,
    batch_size: int,
    max_history: Optional[int],
) -> Dict[str, np.ndarray]:
    node_features, edge_features, time_features, co_features, all_features = [], [], [], [], []

    for start in range(0, data.num_interactions, batch_size):
        end = min(start + batch_size, data.num_interactions)

        src_batch = data.src_node_ids[start:end]
        dst_batch = data.dst_node_ids[start:end]
        ts_batch = data.node_interact_times[start:end]

        src_hist_nodes_list, src_hist_edges_list, src_hist_times_list = sampler.get_all_first_hop_neighbors(
            src_batch, ts_batch
        )
        dst_hist_nodes_list, dst_hist_edges_list, dst_hist_times_list = sampler.get_all_first_hop_neighbors(
            dst_batch, ts_batch
        )

        for src, dst, t, src_nodes, src_edges, src_times, dst_nodes, dst_edges, dst_times in zip(
            src_batch,
            dst_batch,
            ts_batch,
            src_hist_nodes_list,
            src_hist_edges_list,
            src_hist_times_list,
            dst_hist_nodes_list,
            dst_hist_edges_list,
            dst_hist_times_list,
        ):
            src_nodes, src_edges, src_times = maybe_recent_limit(src_nodes, src_edges, src_times, max_history)
            dst_nodes, dst_edges, dst_times = maybe_recent_limit(dst_nodes, dst_edges, dst_times, max_history)

            src_node_ids = np.concatenate([[src], src_nodes]).astype(np.longlong)
            dst_node_ids = np.concatenate([[dst], dst_nodes]).astype(np.longlong)
            node_feat = np.concatenate(
                [
                    summarize_matrix(node_raw[src_node_ids], NODE_FEAT_DIM),
                    summarize_matrix(node_raw[dst_node_ids], NODE_FEAT_DIM),
                ]
            ).astype(np.float32)

            edge_feat = np.concatenate(
                [
                    summarize_matrix(edge_raw[src_edges.astype(np.longlong)], EDGE_FEAT_DIM),
                    summarize_matrix(edge_raw[dst_edges.astype(np.longlong)], EDGE_FEAT_DIM),
                ]
            ).astype(np.float32)

            time_feat = np.concatenate(
                [
                    summarize_time_deltas(t - src_times),
                    summarize_time_deltas(t - dst_times),
                ]
            ).astype(np.float32)

            co_feat = cooccurrence_stats(src_nodes, dst_nodes)
            all_feat = np.concatenate([node_feat, edge_feat, time_feat, co_feat]).astype(np.float32)

            node_features.append(node_feat)
            edge_features.append(edge_feat)
            time_features.append(time_feat)
            co_features.append(co_feat)
            all_features.append(all_feat)

    return {
        "node": np.stack(node_features),
        "edge": np.stack(edge_features),
        "time": np.stack(time_features),
        "co": np.stack(co_features),
        "all": np.stack(all_features),
    }


def pairwise_squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1).unsqueeze(0)
    dist2 = a2 + b2 - 2.0 * (a @ b.t())
    return dist2.clamp_min(0.0)


def estimate_mmd_bandwidth_blocked(
    z: torch.Tensor,
    block_size: int,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> list:
    n = z.size(0)
    if n <= 1:
        base = torch.tensor(1.0, device=z.device, dtype=z.dtype)
    else:
        total = torch.zeros((), device=z.device, dtype=z.dtype)
        for i in range(0, n, block_size):
            zi = z[i : i + block_size]
            for j in range(0, n, block_size):
                zj = z[j : j + block_size]
                dist2 = pairwise_squared_distance(zi, zj)
                total = total + dist2.sum()

        denom = max(n * n - n, 1)
        base = total / float(denom)

    eps = 1e-6
    base = base / (kernel_mul ** (kernel_num // 2))
    return [base * (kernel_mul ** i) + eps for i in range(kernel_num)]


def rbf_kernel_mean_blocked(
    a: torch.Tensor,
    b: torch.Tensor,
    bandwidths: list,
    block_size: int,
) -> float:
    total = 0.0
    count = 0

    for i in range(0, a.size(0), block_size):
        ai = a[i : i + block_size]
        for j in range(0, b.size(0), block_size):
            bj = b[j : j + block_size]
            dist2 = pairwise_squared_distance(ai, bj)

            k_sum = torch.zeros_like(dist2)
            for bandwidth in bandwidths:
                k_sum = k_sum + torch.exp(-dist2 / bandwidth)

            total += float(k_sum.sum().item())
            count += k_sum.numel()

    return total / max(count, 1)


def compute_mmd(
    train_feat: np.ndarray,
    test_feat: np.ndarray,
    device: str,
    block_size: int,
) -> Tuple[float, float]:
    if block_size <= 0:
        raise ValueError("--mmd_block_size must be a positive integer")

    train_n, test_n = normalize_by_train(train_feat, test_feat)
    x = torch.from_numpy(train_n).float().to(device)
    y = torch.from_numpy(test_n).float().to(device)

    with torch.no_grad():
        z = torch.cat([x, y], dim=0)
        bandwidths = estimate_mmd_bandwidth_blocked(z, block_size)
        mean_kxx = rbf_kernel_mean_blocked(x, x, bandwidths, block_size)
        mean_kyy = rbf_kernel_mean_blocked(y, y, bandwidths, block_size)
        mean_kxy = rbf_kernel_mean_blocked(x, y, bandwidths, block_size)
        mmd2 = mean_kxx + mean_kyy - 2.0 * mean_kxy

    return mmd2, float(np.sqrt(max(mmd2, 0.0)))


def analyze_dataset(name: str, cfg: dict, args: argparse.Namespace) -> dict:
    df = pd.read_csv(cfg["path"])
    full_data = build_data_from_df(df)

    train_df = df[(df["ts"] >= cfg["train_ts_start"]) & (df["ts"] <= cfg["train_ts_end"])]
    test_df = df[(df["ts"] >= cfg["test_ts_start"]) & (df["ts"] <= cfg["test_ts_end"])]

    train_data_full = build_data_from_df(train_df)
    test_data_full = build_data_from_df(test_df)
    train_data = sample_data(train_data_full, args.max_interactions, args.seed)
    test_data = sample_data(test_data_full, args.max_interactions, args.seed + 1)

    node_raw, edge_raw = load_raw_features(name)
    sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy="recent",
        seed=args.seed,
    )

    train_feat = extract_interpretable_features(
        train_data, sampler, node_raw, edge_raw, args.batch_size, args.max_history
    )
    test_feat = extract_interpretable_features(
        test_data, sampler, node_raw, edge_raw, args.batch_size, args.max_history
    )

    row = {
        "dataset": name,
        "type": cfg["type"],
        "train_interactions_full": train_data_full.num_interactions,
        "test_interactions_full": test_data_full.num_interactions,
        "train_interactions_used": train_data.num_interactions,
        "test_interactions_used": test_data.num_interactions,
    }

    for key in ["node", "edge", "time", "co", "all"]:
        mmd2, dist = compute_mmd(train_feat[key], test_feat[key], args.device, args.mmd_block_size)
        row[f"MMD_{key}_squared"] = mmd2
        row[f"MMD_{key}_distance"] = dist

    print(
        f"[{name:12s}] "
        f"used={train_data.num_interactions}/{test_data.num_interactions} "
        f"node={row['MMD_node_distance']:.6f} "
        f"edge={row['MMD_edge_distance']:.6f} "
        f"time={row['MMD_time_distance']:.6f} "
        f"co={row['MMD_co_distance']:.6f} "
        f"all={row['MMD_all_distance']:.6f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/test MMD distance using interpretable DyGFormer-style raw historical features."
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_history",
        type=int,
        default=512,
        help="Maximum recent historical neighbors per node; 0 means all history.",
    )
    parser.add_argument(
        "--max_interactions",
        type=int,
        default=5000,
        help="Maximum train/test interactions used per dataset for quadratic MMD; 0 means all.",
    )
    parser.add_argument(
        "--mmd_block_size",
        type=int,
        default=512,
        help="Block size for chunked MMD computation.",
    )
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    selected = {k: v for k, v in DATASETS.items() if k in TARGET_DATASETS}
    rows = [analyze_dataset(name, cfg, args) for name, cfg in selected.items()]
    result = pd.DataFrame(rows).sort_values("MMD_all_distance", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mmd_dygformer_interpretable_raw_stats.csv")
    result.to_csv(out_path, index=False)

    print("\nSummary:")
    print(result.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
