#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画“所有数据集所有时间戳”的新边/旧边数量直方图（堆叠柱状图），并统计平均新边占比。

定义（严格历史口径）：
1) 对每个数据集独立处理，按时间戳升序扫描；
2) 在时间戳 t 的每条边 (u, v)：
   - 若该边在更早时间戳（< t）出现过，则记为旧边；
   - 否则记为新边；
3) 一个时间戳统计该时间戳下边事件数量（非去重）。

输出：
- 图：横轴为“所有数据集的时间戳序列（按数据集拼接）”，纵轴为边数量；
      新边红色，旧边灰色。
- 终端打印：
  a) 所有数据集所有时间戳上，新边占比的简单平均（每个时间戳等权）；
  b) 全局加权新边占比（总新边 / 总边）。
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "wikipedia",
    "uci",
    "CanParl",
    "USLegis",
    "mooc",
    "enron",
    "reddit",
    "UNtrade",
    "UNvote",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram of new/old edge counts for all timestamps across datasets."
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS, help="数据集列表")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="与现有 DataLoader 保持一致的 val_ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="与现有 DataLoader 保持一致的 test_ratio")
    parser.add_argument(
        "--output_path",
        type=str,
        default="visualizations/new_old_edges_by_timestamp_all9.png",
        help="输出图像路径",
    )
    parser.add_argument("--dpi", type=int, default=300, help="输出图像 DPI")
    parser.add_argument(
        "--fig_width",
        type=float,
        default=18.0,
        help="图宽（英寸）",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=6.0,
        help="图高（英寸）",
    )
    parser.add_argument(
        "--save_stats_csv",
        type=str,
        default="visualizations/new_old_edges_by_timestamp_all9_stats.csv",
        help="可选：保存每个（数据集, 时间戳）统计结果",
    )
    parser.add_argument(
        "--max_bins_per_dataset",
        type=int,
        default=100,
        help="每个数据集时间轴最多分几箱（避免百万根柱子卡死），0 表示只画每数据集一根汇总柱",
    )
    return parser.parse_args()


def _to_directed_edge(src: int, dst: int) -> Tuple[int, int]:
    return int(src), int(dst)


def _resolve_dataset_csv_path(dataset_name: str) -> str:
    candidates = [
        os.path.join("processed_data", dataset_name, "ml_{}.csv".format(dataset_name)),
        os.path.join("..", "processed_data", dataset_name, "ml_{}.csv".format(dataset_name)),
        os.path.join("DG_data", dataset_name, "ml_{}.csv".format(dataset_name)),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "找不到数据集 {} 的 CSV，已尝试路径: {}".format(dataset_name, candidates)
    )


def compute_dataset_timestamp_counts(
    dataset_name: str,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, np.ndarray]:
    _ = (val_ratio, test_ratio)  # 保留参数以兼容现有调用，不参与计算
    csv_path = _resolve_dataset_csv_path(dataset_name)
    df = pd.read_csv(csv_path)
    required_cols = ["u", "i", "ts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("{} 缺少列: {}".format(csv_path, missing))

    src_all = pd.to_numeric(df["u"], errors="coerce").to_numpy()
    dst_all = pd.to_numeric(df["i"], errors="coerce").to_numpy()
    ts_all = pd.to_numeric(df["ts"], errors="coerce").to_numpy()
    valid_mask = np.isfinite(src_all) & np.isfinite(dst_all) & np.isfinite(ts_all)
    src_all = src_all[valid_mask].astype(np.int64)
    dst_all = dst_all[valid_mask].astype(np.int64)
    ts_all = ts_all[valid_mask].astype(np.float64)

    order = np.argsort(ts_all, kind="stable")
    src_all = src_all[order]
    dst_all = dst_all[order]
    ts_all = ts_all[order]

    unique_timestamps: List[float] = []
    new_counts: List[int] = []
    old_counts: List[int] = []
    ratios: List[float] = []

    seen_edges = set()
    n = len(ts_all)
    i = 0
    while i < n:
        cur_ts = ts_all[i]
        j = i
        while j < n and ts_all[j] == cur_ts:
            j += 1

        cur_new = 0
        cur_old = 0
        cur_batch_edges: List[Tuple[int, int]] = []
        for k in range(i, j):
            edge = _to_directed_edge(src_all[k], dst_all[k])
            cur_batch_edges.append(edge)
            if edge in seen_edges:
                cur_old += 1
            else:
                cur_new += 1

        for edge in cur_batch_edges:
            seen_edges.add(edge)

        total = cur_new + cur_old
        ratio = float(cur_new / total) if total > 0 else np.nan

        unique_timestamps.append(float(cur_ts))
        new_counts.append(int(cur_new))
        old_counts.append(int(cur_old))
        ratios.append(ratio)
        i = j

    return {
        "timestamps": np.asarray(unique_timestamps, dtype=np.float64),
        "new_counts": np.asarray(new_counts, dtype=np.int64),
        "old_counts": np.asarray(old_counts, dtype=np.int64),
        "new_ratio": np.asarray(ratios, dtype=np.float64),
    }


def main() -> None:
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    all_new_counts: List[int] = []
    all_old_counts: List[int] = []
    all_ratios: List[float] = []
    x_labels: List[str] = []
    dataset_boundaries: List[int] = []
    stats_rows: List[Dict[str, float]] = []

    cursor = 0
    for ds in args.datasets:
        print(f"\n[统计] {ds}")
        stats = compute_dataset_timestamp_counts(
            dataset_name=ds,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        ts = stats["timestamps"]
        new_c = stats["new_counts"]
        old_c = stats["old_counts"]
        ratio = stats["new_ratio"]

        for t, n_new, n_old, r in zip(ts, new_c, old_c, ratio):
            all_new_counts.append(int(n_new))
            all_old_counts.append(int(n_old))
            all_ratios.append(float(r))
            x_labels.append(f"{ds}:{t:g}")
            stats_rows.append(
                {
                    "dataset": ds,
                    "timestamp": float(t),
                    "new_edges": int(n_new),
                    "old_edges": int(n_old),
                    "total_edges": int(n_new + n_old),
                    "new_ratio": float(r),
                }
            )

        cursor += len(ts)
        dataset_boundaries.append(cursor)
        print(
            f"  时间戳数={len(ts)}, 总边数={int(np.sum(new_c + old_c))}, "
            f"新边总数={int(np.sum(new_c))}, 旧边总数={int(np.sum(old_c))}"
        )

    if len(all_new_counts) == 0:
        raise RuntimeError("没有可统计的数据，请检查输入数据集。")

    new_arr = np.asarray(all_new_counts, dtype=np.int64)
    old_arr = np.asarray(all_old_counts, dtype=np.int64)
    ratio_arr = np.asarray(all_ratios, dtype=np.float64)

    # 先输出统计，不依赖画图
    mean_new_ratio_per_timestamp = float(np.nanmean(ratio_arr))
    weighted_new_ratio = float(np.sum(new_arr) / np.sum(new_arr + old_arr))
    print("\n" + "=" * 60)
    print("所有数据集、所有时间戳的统计：")
    print("  平均新边占比（每个时间戳等权）: {:.6f}".format(mean_new_ratio_per_timestamp))
    print("  全局新边占比（按边数量加权）:   {:.6f}".format(weighted_new_ratio))
    print("=" * 60)

    # 图：按时间分箱或按数据集汇总，避免百万根柱子卡死
    max_bins = args.max_bins_per_dataset
    if max_bins is None or max_bins <= 0:
        max_bins = 0

    if max_bins == 0:
        # 每个数据集一根堆叠柱
        num_ds = len(args.datasets)
        new_per_ds = []
        old_per_ds = []
        start = 0
        for end in dataset_boundaries:
            new_per_ds.append(int(np.sum(new_arr[start:end])))
            old_per_ds.append(int(np.sum(old_arr[start:end])))
            start = end
        plot_new = np.array(new_per_ds, dtype=np.int64)
        plot_old = np.array(old_per_ds, dtype=np.int64)
        plot_boundaries = np.arange(num_ds + 1)
        x_pos = np.arange(num_ds)
        fig_w = max(10, num_ds * 1.2)
        title = "New vs Old Edge Counts by Dataset (aggregated)"
    else:
        # 每个数据集内按时间分成最多 max_bins 箱，每箱一根堆叠柱
        plot_new_list = []
        plot_old_list = []
        plot_boundaries = [0]
        segment_names = []
        start = 0
        for ds, end in zip(args.datasets, dataset_boundaries):
            n_ts = end - start
            new_slice = new_arr[start:end]
            old_slice = old_arr[start:end]
            n_bins = min(max_bins, n_ts)
            start = end
            if n_bins <= 0:
                continue
            # 按索引均分为 n_bins 组，每组内求和
            bin_edges = np.linspace(0, n_ts, n_bins + 1, dtype=np.int64)
            new_binned = np.zeros(n_bins, dtype=np.int64)
            old_binned = np.zeros(n_bins, dtype=np.int64)
            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                new_binned[b] = int(np.sum(new_slice[lo:hi]))
                old_binned[b] = int(np.sum(old_slice[lo:hi]))
            plot_new_list.append(new_binned)
            plot_old_list.append(old_binned)
            plot_boundaries.append(plot_boundaries[-1] + n_bins)
            segment_names.append(ds)
        plot_new = np.concatenate(plot_new_list) if plot_new_list else np.array([], dtype=np.int64)
        plot_old = np.concatenate(plot_old_list) if plot_old_list else np.array([], dtype=np.int64)
        n_bars = len(plot_new)
        x_pos = np.arange(n_bars)
        fig_w = min(24, max(12, n_bars * 0.03))
        title = "New vs Old Edge Counts by Time (binned, {} bins per dataset max)".format(max_bins)

    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar(x_pos, plot_new, color="#d62728", width=1.0, label="New edges")
    ax.bar(x_pos, plot_old, bottom=plot_new, color="#9e9e9e", width=1.0, label="Old edges")
    if len(plot_boundaries) > 2:
        for b in plot_boundaries[1:-1]:
            ax.axvline(x=b - 0.5, color="black", linewidth=0.5, alpha=0.4)
    if max_bins == 0:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(args.datasets, rotation=30, ha="right")
    else:
        tick_pos = [(plot_boundaries[i] + plot_boundaries[i + 1] - 1) / 2.0 for i in range(len(plot_boundaries) - 1)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(segment_names, rotation=0)
    ax.set_ylabel("Number of edges")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output_path, dpi=args.dpi)
    plt.close()
    print("图已保存: {}".format(args.output_path))

    if args.save_stats_csv:
        csv_dir = os.path.dirname(args.save_stats_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        pd.DataFrame(stats_rows).to_csv(args.save_stats_csv, index=False)
        print("逐时间戳统计已保存: {}".format(args.save_stats_csv))


if __name__ == "__main__":
    main()
