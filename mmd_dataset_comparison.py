# -*- coding: utf-8 -*-
"""
在节点级原始特征上比较 train / test 分布差异。

节点特征（4 维，来自有向图）：
  [weighted_indegree, weighted_outdegree,
   avg_neigh_weighted_indeg, avg_neigh_weighted_outdeg]

仅使用经典 MMD（二次复杂度、多带宽 RBF 核，与 NForGOT.General_MMD_loss 同型，
但无硬编码维度），在训练集与测试集上使用该时间窗内**全部节点**的特征。

可选：--max_nodes 在样本过多时子采样以防 OOM。
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from collections import Counter
from typing import Dict, Any, Tuple, Optional


# ============================== 数据集配置 ==============================
DATASETS = {
    "USLegis":   {"path": "processed_data/USLegis/ml_USLegis.csv",   "train_ts_start": 0.0,          "train_ts_end": 8.0,            "test_ts_start": 10.0,           "test_ts_end": 11.0,            "type": "political"},
    "CanParl":   {"path": "processed_data/CanParl/ml_CanParl.csv",   "train_ts_start": 0.0,          "train_ts_end": 283996800.0,    "test_ts_start": 347155200.0,    "test_ts_end": 410227200.0,     "type": "political"},
    "UNvote":    {"path": "processed_data/UNvote/ml_UNvote.csv",     "train_ts_start": 0.0,          "train_ts_end": 1830297600.0,   "test_ts_start": 2019686400.0,   "test_ts_end": 2240611200.0,    "type": "political"},
    "UNtrade":   {"path": "processed_data/UNtrade/ml_UNtrade.csv",   "train_ts_start": 0.0,          "train_ts_end": 757382400.0,    "test_ts_start": 883612800.0,    "test_ts_end": 978307200.0,     "type": "political"},
    "wikipedia": {"path": "processed_data/wikipedia/ml_wikipedia.csv","train_ts_start": 0.0,         "train_ts_end": 1862645.0,      "test_ts_start": 2218288.0,      "test_ts_end": 2678373.0,       "type": "social"},
    "reddit":    {"path": "processed_data/reddit/ml_reddit.csv",     "train_ts_start": 0.0,          "train_ts_end": 1882467.555,    "test_ts_start": 2261811.988,    "test_ts_end": 2678390.016,     "type": "social"},
    "uci":       {"path": "processed_data/uci/ml_uci.csv",           "train_ts_start": 0.0,          "train_ts_end": 3834783.0,      "test_ts_start": 6714637.0,      "test_ts_end": 16736181.0,      "type": "social"},
    "enron":     {"path": "processed_data/enron/ml_enron.csv",       "train_ts_start": 0.0,          "train_ts_end": 83842852.0,     "test_ts_start": 93431801.0,     "test_ts_end": 113740399.0,     "type": "social"},
    "mooc":      {"path": "processed_data/mooc/ml_mooc.csv",         "train_ts_start": 0.0,          "train_ts_end": 1917226.0,      "test_ts_start": 2250152.0,      "test_ts_end": 2572086.0,       "type": "mooc"},
}


def _register_lastfm(datasets: Dict[str, Dict[str, Any]],
                     path: str = "processed_data/lastfm/ml_lastfm.csv",
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> None:
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    if "ts" not in df.columns or df["ts"].size == 0:
        return
    ts = df["ts"].to_numpy(dtype=np.float64)
    val_time, test_time = np.quantile(ts, [1.0 - val_ratio - test_ratio, 1.0 - test_ratio])
    datasets["lastfm"] = {
        "path": path,
        "train_ts_start": 0.0,
        "train_ts_end": float(val_time),
        "test_ts_start": float(test_time),
        "test_ts_end": float(ts.max()),
        "type": "social",
    }


# ============================== 经典 MMD（多带宽 RBF，支持 n≠m） ==============================
class ClassicMMDLoss(nn.Module):
    """
    经典（二次）MMD：拼接 Z = [X; Y]，多带宽 RBF 核矩阵 K，
    有偏估计：MMD^2 ≈ mean(K_xx) + mean(K_yy) - 2 * mean(K_xy)
    与 General_MMD_loss 同型，但不对特征做 reshape 硬编码。
    """
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _multi_rbf_kernel(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: (N, d) -> K: (N, N)，多带宽 RBF 之和。"""
        n = Z.size(0)
        if n <= 1:
            return torch.zeros(n, n, device=Z.device, dtype=Z.dtype)

        eps = 1e-6
        # pairwise squared L2: (N, N)
        z0 = Z.unsqueeze(0).expand(n, n, -1)
        z1 = Z.unsqueeze(1).expand(n, n, -1)
        l2 = ((z0 - z1) ** 2).sum(dim=2)

        denom = max(n * n - n, 1)
        bandwidth = l2.detach().sum() / float(denom)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bw_list = [bandwidth * (self.kernel_mul ** i) + eps for i in range(self.kernel_num)]

        k_sum = torch.zeros_like(l2)
        for b in bw_list:
            k_sum = k_sum + torch.exp(-l2 / b)
        return k_sum

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (n, d), Y: (m, d)。返回标量 MMD^2（有偏估计）。
        """
        n, m = X.size(0), Y.size(0)
        if n < 1 or m < 1:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)

        Z = torch.cat([X, Y], dim=0)
        K = self._multi_rbf_kernel(Z)
        K_xx = K[:n, :n]
        K_yy = K[n:, n:]
        K_xy = K[:n, n:]
        return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()


# ============================== 图构建 & 节点特征 ==============================
def build_directed_graph(df: pd.DataFrame) -> nx.DiGraph:
    counts = Counter(zip(df["u"], df["i"]))
    G = nx.DiGraph()
    for (u, v), w in counts.items():
        if u != v:
            G.add_edge(u, v, weight=float(w))
    return G


def extract_node_feature_matrix(G: nx.DiGraph) -> np.ndarray:
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    in_w = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))
    rows = []
    for u in nodes:
        neighbors = list(G.neighbors(u))
        if neighbors:
            avg_n_indeg = float(np.mean([in_w.get(v, 0.0) for v in neighbors]))
            avg_n_outdeg = float(np.mean([out_w.get(v, 0.0) for v in neighbors]))
        else:
            avg_n_indeg = avg_n_outdeg = 0.0
        rows.append([
            float(in_w.get(u, 0.0)),
            float(out_w.get(u, 0.0)),
            avg_n_indeg,
            avg_n_outdeg,
        ])
    return np.array(rows, dtype=np.float32)


def maybe_subsample(
    feat_train: np.ndarray,
    feat_test: np.ndarray,
    max_nodes: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """若 max_nodes 非空，从两侧各最多采 max_nodes 条（不放回）。"""
    n0, m0 = len(feat_train), len(feat_test)
    if max_nodes is None or max_nodes <= 0:
        return feat_train, feat_test, n0, m0
    rng = np.random.default_rng(seed)
    nt = min(n0, max_nodes)
    mt = min(m0, max_nodes)
    idx_tr = rng.choice(n0, size=nt, replace=False) if n0 > nt else np.arange(n0)
    idx_te = rng.choice(m0, size=mt, replace=False) if m0 > mt else np.arange(m0)
    return feat_train[idx_tr], feat_test[idx_te], n0, m0


def compute_mmd_value(feat_train: np.ndarray, feat_test: np.ndarray, device: str) -> float:
    x = torch.from_numpy(feat_train).float().to(device)
    y = torch.from_numpy(feat_test).float().to(device)
    fn = ClassicMMDLoss().to(device)
    with torch.no_grad():
        return float(fn(x, y).item())


# ============================== 主分析 ==============================
def analyze_dataset_mmd(
    name: str,
    cfg: Dict[str, Any],
    max_nodes: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    try:
        df = pd.read_csv(cfg["path"])[["u", "i", "ts"]]
        train_df = df[(df["ts"] >= cfg["train_ts_start"]) & (df["ts"] <= cfg["train_ts_end"])]
        test_df = df[(df["ts"] >= cfg["test_ts_start"]) & (df["ts"] <= cfg["test_ts_end"])]

        G_train = build_directed_graph(train_df)
        G_test = build_directed_graph(test_df)

        feat_train = extract_node_feature_matrix(G_train)
        feat_test = extract_node_feature_matrix(G_test)

        col_min = feat_train.min(axis=0)
        col_max = feat_train.max(axis=0)
        eps = 1e-8
        feat_train_n = (feat_train - col_min) / np.maximum(col_max - col_min, eps)
        feat_test_n = (feat_test - col_min) / np.maximum(col_max - col_min, eps)

        f_tr, f_te, n_full, m_full = maybe_subsample(feat_train_n, feat_test_n, max_nodes, seed)

        if len(f_tr) < 1 or len(f_te) < 1:
            raise ValueError("train 或 test 侧无节点")

        mmd_val = compute_mmd_value(f_tr, f_te, device)

        used_tr, used_te = len(f_tr), len(f_te)
        sub_note = ""
        if max_nodes and (n_full > max_nodes or m_full > max_nodes):
            sub_note = f" (子采样用于计算: train≤{max_nodes}, test≤{max_nodes})"

        print(
            f"[{name:12s}] train_nodes={n_full:6d} test_nodes={m_full:6d}  "
            f"used={used_tr}/{used_te}  MMD²={mmd_val:.8f}{sub_note}"
        )

        return {
            "dataset": name,
            "type": cfg["type"],
            "train_nodes_full": n_full,
            "test_nodes_full": m_full,
            "train_nodes_used": used_tr,
            "test_nodes_used": used_te,
            "train_edges": len(train_df),
            "test_edges": len(test_df),
            "MMD_squared": mmd_val,
        }
    except Exception as e:
        print(f"[{name}] 出错: {e}")
        return {
            "dataset": name,
            "type": cfg.get("type", "unknown"),
            "train_nodes_full": 0,
            "test_nodes_full": 0,
            "train_nodes_used": 0,
            "test_nodes_used": 0,
            "train_edges": 0,
            "test_edges": 0,
            "MMD_squared": float("nan"),
            "error": str(e),
        }


def analyze_all(
    datasets: Dict[str, Dict[str, Any]],
    max_nodes: Optional[int],
    device: str,
    seed: int,
) -> pd.DataFrame:
    print("=" * 72)
    print(
        f"经典 MMD²（有偏估计）  |  "
        f"{'全部节点' if not max_nodes else f'max_nodes={max_nodes} 子采样'}  |  device={device}"
    )
    print(
        "节点特征: [weighted_indeg, weighted_outdeg, "
        "avg_neigh_weighted_indeg, avg_neigh_weighted_outdeg]"
    )
    print("=" * 72)
    rows = [
        analyze_dataset_mmd(name, cfg, max_nodes, device, seed)
        for name, cfg in datasets.items()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vs test 节点分布经典 MMD²")
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=0,
        help="每侧最多使用的节点数；0 表示不限制（用该时间窗内全部节点，可能占大量显存/内存）",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    max_n = None if args.max_nodes <= 0 else args.max_nodes

    _register_lastfm(DATASETS)
    if "lastfm" not in DATASETS:
        print("提示: 未找到 lastfm，已跳过注册")

    result_df = analyze_all(DATASETS, max_n, args.device, args.seed)

    print("\n" + "=" * 72)
    print("汇总（按 MMD² 降序）")
    print("=" * 72)
    cols = ["dataset", "type", "train_nodes_full", "test_nodes_full",
            "train_nodes_used", "test_nodes_used", "MMD_squared"]
    summary = result_df[[c for c in cols if c in result_df.columns]].sort_values(
        "MMD_squared", ascending=False, na_position="last"
    )
    print(summary.to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mmd_train_test_comparison.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")
