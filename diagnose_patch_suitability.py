#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计九个数据集中“同一时间戳历史邻居占比”的节点平均值。

定义（按节点侧，即每条交互的 src 与 dst 都各算一次）：
1) 对当前节点在当前时间戳 t 的一次出现，只看严格早于 t 的历史交互；
2) 该节点历史中，按时间戳分组统计每个时间戳对应的历史邻居数量；
3) 比例 = max(单一时间戳下的历史邻居数量) / 全部历史邻居数量；
4) 对一个数据集内所有节点侧样本的该比例取平均。
"""

import argparse
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.DataLoader import get_link_prediction_data


DATASETS = [
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


def finalize_node_ratio(node_hist_total: int, node_hist_time_counter: Dict[float, int]) -> float:
    """计算单个节点在某次出现时的比例。"""
    if node_hist_total <= 0:
        return np.nan
    max_same_ts_hist_neighbors = max(node_hist_time_counter.values())
    return float(max_same_ts_hist_neighbors / node_hist_total)


def update_history_for_one_interaction(
    src: int,
    dst: int,
    ts: float,
    hist_total: DefaultDict[int, int],
    hist_time_counter: DefaultDict[int, DefaultDict[float, int]],
) -> None:
    """把当前交互写入历史（在完成当前时间戳全部样本的计算之后调用）。"""
    hist_total[src] += 1
    hist_total[dst] += 1
    hist_time_counter[src][ts] += 1
    hist_time_counter[dst][ts] += 1


def compute_dataset_ratio(dataset_name: str, val_ratio: float, test_ratio: float) -> Dict[str, float]:
    """按“严格历史”口径计算数据集统计量。"""
    _, _, full_data, _, _, _, _, _ = get_link_prediction_data(
        dataset_name=dataset_name,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    src_all = np.asarray(full_data.src_node_ids, dtype=np.int64)
    dst_all = np.asarray(full_data.dst_node_ids, dtype=np.int64)
    ts_all = np.asarray(full_data.node_interact_times)

    # 先按时间升序，确保“历史”定义正确。
    order = np.argsort(ts_all, kind="stable")
    src_all = src_all[order]
    dst_all = dst_all[order]
    ts_all = ts_all[order]

    hist_total: DefaultDict[int, int] = defaultdict(int)
    hist_time_counter: DefaultDict[int, DefaultDict[float, int]] = defaultdict(lambda: defaultdict(int))
    ratios: List[float] = []
    num_no_history = 0

    n = len(ts_all)
    i = 0
    while i < n:
        cur_ts = ts_all[i]
        j = i
        while j < n and ts_all[j] == cur_ts:
            j += 1

        # 当前时间戳批次先计算（只使用 < cur_ts 的历史），再统一写入历史。
        for k in range(i, j):
            src = int(src_all[k])
            dst = int(dst_all[k])

            src_ratio = finalize_node_ratio(hist_total[src], hist_time_counter[src])
            dst_ratio = finalize_node_ratio(hist_total[dst], hist_time_counter[dst])

            if np.isnan(src_ratio):
                num_no_history += 1
            else:
                ratios.append(src_ratio)

            if np.isnan(dst_ratio):
                num_no_history += 1
            else:
                ratios.append(dst_ratio)

        for k in range(i, j):
            update_history_for_one_interaction(
                src=int(src_all[k]),
                dst=int(dst_all[k]),
                ts=float(cur_ts),
                hist_total=hist_total,
                hist_time_counter=hist_time_counter,
            )
        i = j

    mean_ratio = float(np.mean(ratios)) if len(ratios) > 0 else np.nan
    return {
        "dataset": dataset_name,
        "num_interactions": int(n),
        "num_node_sides": int(2 * n),
        "num_with_history": int(len(ratios)),
        "num_without_history": int(num_no_history),
        "mean_max_same_timestamp_ratio": mean_ratio,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计九个数据集的同一时间戳历史邻居占比")
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS, help="要统计的数据集列表")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="temporal_same_timestamp_neighbor_ratio.csv",
        help="输出结果 CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    rows = []
    print("=" * 90)
    print("同一时间戳历史邻居占比统计（严格历史: 仅使用当前时间戳之前交互）")
    print("=" * 90)

    for ds in args.datasets:
        print(f"\n[统计] {ds}")
        try:
            row = compute_dataset_ratio(ds, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
            rows.append(row)
            print(
                f"  mean_max_same_timestamp_ratio={row['mean_max_same_timestamp_ratio']:.6f}, "
                f"with_history={row['num_with_history']}, without_history={row['num_without_history']}"
            )
        except Exception as exc:
            print(f"  跳过 {ds}（异常: {exc}）")
            rows.append(
                {
                    "dataset": ds,
                    "num_interactions": 0,
                    "num_node_sides": 0,
                    "num_with_history": 0,
                    "num_without_history": 0,
                    "mean_max_same_timestamp_ratio": np.nan,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\n结果已保存: {args.output_csv}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print("\n统计摘要：")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
