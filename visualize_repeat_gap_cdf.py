#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画“节点对连续重复交互时间间隔”的 CDF（x 轴 log）。

定义：
1) 以节点对 (u, v) 分组（默认按有向对；可选无向）；
2) 对每个节点对按时间升序排序；
3) 对连续出现计算 gap = t_k - t_{k-1}；
4) 汇总所有节点对的 gap，画经验 CDF: P(gap <= x)。

示例：
python visualize_repeat_gap_cdf.py --datasets wikipedia uci mooc
python visualize_repeat_gap_cdf.py --csv_paths /path/a.csv /path/b.csv --labels A B
python visualize_repeat_gap_cdf.py --datasets enron --undirected --output_path repeat_gap_cdf.png
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        description="Plot log-x CDF of consecutive repeat interaction gaps for node pairs."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="数据集名（将自动查找 processed_data/<ds>/ml_<ds>.csv）",
    )
    parser.add_argument(
        "--csv_paths",
        type=str,
        nargs="+",
        default=None,
        help="直接指定一个或多个 CSV 路径（需包含列 u,i,ts）",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="可选：与 --csv_paths 一一对应的图例名称",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="将 (u,v) 与 (v,u) 视为同一节点对",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default="ts",
        help="时间列名，默认 ts",
    )
    parser.add_argument(
        "--src_col",
        type=str,
        default="u",
        help="源节点列名，默认 u",
    )
    parser.add_argument(
        "--dst_col",
        type=str,
        default="i",
        help="目标节点列名，默认 i",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="将 gap<=0 映射到 eps，以便 log 轴显示",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="visualizations/repeat_gap_cdf_logx.png",
        help="输出图像路径",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图像 DPI",
    )
    return parser.parse_args()


def _candidate_paths(dataset_name: str) -> List[str]:
    return [
        os.path.join("processed_data", dataset_name, f"ml_{dataset_name}.csv"),
        os.path.join("..", "processed_data", dataset_name, f"ml_{dataset_name}.csv"),
        os.path.join("DG_data", dataset_name, f"ml_{dataset_name}.csv"),
    ]


def resolve_dataset_csv(dataset_name: str) -> str:
    for p in _candidate_paths(dataset_name):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"找不到数据集 {dataset_name} 的 CSV。已尝试: {_candidate_paths(dataset_name)}"
    )


def load_inputs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []

    if args.csv_paths:
        labels = args.labels if args.labels else [os.path.splitext(os.path.basename(p))[0] for p in args.csv_paths]
        if len(labels) != len(args.csv_paths):
            raise ValueError("--labels 数量必须与 --csv_paths 一致")
        for lbl, p in zip(labels, args.csv_paths):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"CSV 不存在: {p}")
            items.append((lbl, p))
        return items

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    for ds in datasets:
        items.append((ds, resolve_dataset_csv(ds)))
    return items


def compute_pair_gaps(
    csv_path: str,
    src_col: str,
    dst_col: str,
    time_col: str,
    undirected: bool,
    eps: float,
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    missing = [c for c in [src_col, dst_col, time_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    df = df[[src_col, dst_col, time_col]].copy()
    df[src_col] = pd.to_numeric(df[src_col], errors="coerce")
    df[dst_col] = pd.to_numeric(df[dst_col], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[src_col, dst_col, time_col])

    if undirected:
        u = np.minimum(df[src_col].to_numpy(), df[dst_col].to_numpy())
        v = np.maximum(df[src_col].to_numpy(), df[dst_col].to_numpy())
        df["_u"] = u
        df["_v"] = v
        pair_cols = ["_u", "_v"]
    else:
        pair_cols = [src_col, dst_col]

    df = df.sort_values(pair_cols + [time_col], kind="stable")
    df["gap"] = df.groupby(pair_cols, sort=False)[time_col].diff()
    gaps = df["gap"].dropna().to_numpy(dtype=np.float64)
    if gaps.size == 0:
        return gaps

    # 为了 log 轴可视化，非正 gap 收敛到 eps
    gaps = np.where(gaps <= 0.0, eps, gaps)
    return gaps


def empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / float(len(x))
    return x, y


def main() -> None:
    args = parse_args()
    items = load_inputs(args)

    plt.figure(figsize=(9, 6))
    total_curves = 0

    for label, path in items:
        gaps = compute_pair_gaps(
            csv_path=path,
            src_col=args.src_col,
            dst_col=args.dst_col,
            time_col=args.time_col,
            undirected=args.undirected,
            eps=args.eps,
        )
        if gaps.size == 0:
            print(f"[跳过] {label}: 无连续重复交互（没有可计算的 gap）")
            continue

        x, y = empirical_cdf(gaps)
        plt.plot(x, y, linewidth=2, label=f"{label} (n={len(gaps)})")
        total_curves += 1

        print(
            f"[完成] {label}: num_gaps={len(gaps)}, min={np.min(gaps):.3e}, "
            f"median={np.median(gaps):.3e}, p90={np.percentile(gaps, 90):.3e}"
        )

    if total_curves == 0:
        raise RuntimeError("没有可绘制的曲线，请检查输入 CSV 是否包含重复节点对。")

    plt.xscale("log")
    plt.xlabel("Consecutive repeat gap Δt (log scale)")
    plt.ylabel("CDF: P(gap <= Δt)")
    title_suffix = "undirected pairs" if args.undirected else "directed pairs"
    plt.title(f"Repeat Interaction Gap CDF ({title_suffix})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output_path, dpi=args.dpi)
    print(f"\n图已保存: {args.output_path}")


if __name__ == "__main__":
    main()
