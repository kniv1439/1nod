#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18

# Radar axes: 9 datasets
labels = [
    "Wikipedia", "Reddit", "UCI", "Enron", "MOOC",
    "UN Trade", "UN Vote", "US Legis.", "Can.Parl."
]

# 4 metrics (2 settings x 2 metrics), each gets its own subplot
metrics = ["trans_AP", "trans_AUC", "ind_AP", "ind_AUC"]

# Subplot display titles corresponding to each metric key
metric_titles = {
    "trans_AP":  "Transductive AP",
    "trans_AUC": "Transductive AUC-ROC",
    "ind_AP":    "Inductive AP",
    "ind_AUC":   "Inductive AUC-ROC",
}

# Model data: model_name -> {metric: [value_per_dataset x9]}
# Replace with actual values
models = {
    # data order matches labels: Wikipedia, Reddit, Uci, Enron, Mooc, UNtrade, UNvote, USLegis, CanParl
    # NOTE: LaTeX table column order is Wikipedia,Reddit,UCI,Enron,Mooc,CanParl,USLegis,UNTrade,UNVote
    #       Values here are reordered to match Python label order above.
    "CASTDyG": {
        "trans_AP":  [99.18, 99.41, 97.02, 93.28, 89.27, 68.28, 57.73, 73.61, 95.55],
        "trans_AUC": [99.07, 99.22, 96.53, 93.64, 88.76, 70.95, 59.41, 79.84, 96.84],
        "ind_AP":    [98.59, 99.14, 95.27, 90.75, 88.30, 67.52, 57.21, 57.90, 90.88],
        "ind_AUC":   [98.46, 98.79, 93.14, 90.58, 88.00, 68.84, 57.47, 56.02, 92.92],
    },
    "CARoDyG": {
        "trans_AP":  [99.06, 99.37, 96.85, 93.23, 89.14, 67.89, 56.94, 73.28, 95.47],
        "trans_AUC": [98.96, 99.20, 96.39, 93.17, 88.64, 70.72, 58.75, 79.24, 96.77],
        "ind_AP":    [98.28, 99.11, 95.31, 90.57, 88.11, 67.34, 56.89, 57.73, 90.83],
        "ind_AUC":   [98.12, 98.75, 92.87, 90.48, 87.43, 68.81, 55.38, 54.44, 92.80],
    },
    "DASTDyG": {
        "trans_AP":  [99.32, 99.39, 97.53, 96.88, 95.97, 94.97, 85.67, 85.27, 98.05],
        "trans_AUC": [99.24, 99.13, 97.19, 96.02, 95.54, 94.92, 85.03, 79.56, 98.11],
        "ind_AP":    [98.96, 99.10, 96.01, 92.66, 94.72, 92.88, 67.24, 68.82, 96.42],
        "ind_AUC":   [98.84, 98.73, 95.27, 91.25, 92.48, 91.80, 66.95, 56.33, 95.94],
    },
    "DARoDyG": {
        "trans_AP":  [99.12, 99.30, 97.01, 96.24, 95.76, 94.92, 85.08, 85.12, 97.86],
        "trans_AUC": [99.01, 99.09, 97.12, 95.87, 95.12, 94.86, 84.91, 78.36, 97.96],
        "ind_AP":    [98.56, 99.06, 95.64, 92.44, 94.53, 92.85, 66.22, 68.67, 96.14],
        "ind_AUC":   [98.52, 98.67, 95.23, 90.96, 92.18, 91.68, 66.87, 56.07, 95.82],
    },
    "RepeatMixer": {
        "trans_AP":  [99.16, 99.22, 96.74, 92.66, 92.76, 64.00, 55.33, 68.36, 81.30],
        "trans_AUC": [99.04, 99.15, 95.36, 93.47, 93.60, 69.92, 56.58, 75.07, 86.15],
        "ind_AP":    [98.70, 98.85, 95.04, 87.97, 93.05, 61.36, 54.47, 51.70, 59.77],
        "ind_AUC":   [98.63, 98.69, 93.54, 89.04, 94.10, 66.17, 54.32, 49.29, 60.03],
    },
    "TIDFormer": {
        "trans_AP":  [99.30, 99.38, 97.44, 93.17, 93.42, 60.84, 69.01, 66.51, 99.84],
        "trans_AUC": [99.19, 98.93, 96.64, 92.72, 92.79, 61.25, 67.43, 71.81, 99.87],
        "ind_AP":    [98.93, 98.98, 95.95, 89.92, 93.29, 60.31, 66.02, 60.83, 93.97],
        "ind_AUC":   [98.76, 98.23, 93.35, 85.62, 92.55, 61.14, 65.13, 59.55, 94.52],
    },
    "DyGFormer": {
        "trans_AP":  [98.81, 98.82, 95.51, 91.33, 86.48, 66.46, 55.55, 71.11, 97.36],
        "trans_AUC": [98.64, 98.63, 94.01, 91.11, 86.29, 70.20, 57.12, 77.90, 97.76],
        "ind_AP":    [98.36, 98.35, 94.14, 88.63, 85.27, 64.55, 55.93, 54.28, 87.74],
        "ind_AUC":   [98.16, 98.05, 91.97, 88.56, 85.00, 67.25, 56.73, 53.21, 89.33],
    },
    "CNEN": {
        "trans_AP":  [99.09, 99.22, 96.85, 92.48, 94.18, 76.92, 66.40, 78.69, 86.09],
        "trans_AUC": [99.01, 99.15, 96.03, 93.12, 95.69, 78.57, 69.52, 84.14, 89.71],
        "ind_AP":    [98.37, 98.78, 95.03, 89.66, 91.89, 66.58, 69.71, 59.44, 68.31],
        "ind_AUC":   [98.23, 98.62, 93.34, 90.18, 92.76, 67.26, 69.17, 61.52, 70.22],
    },
    "TGAT": {
        "trans_AP":  [96.94, 98.52, 79.63, 71.12, 85.84, 61.47, 52.21, 68.52, 70.73],
        "trans_AUC": [96.67, 98.47, 78.53, 68.89, 87.11, 64.01, 52.83, 75.84, 75.69],
        "ind_AP":    [96.22, 97.09, 79.54, 67.05, 85.50, 61.03, 52.24, 51.00, 55.18],
        "ind_AUC":   [95.90, 96.98, 77.64, 64.63, 86.84, 62.72, 51.83, 48.27, 56.51],
    },
    "TGN": {
        "trans_AP":  [98.45, 98.63, 92.34, 86.53, 98.15, 65.03, 65.72, 75.99, 70.88],
        "trans_AUC": [98.37, 98.60, 92.03, 88.32, 91.21, 69.10, 69.71, 83.34, 76.99],
        "ind_AP":    [97.83, 97.50, 88.12, 77.94, 89.04, 58.31, 58.85, 58.63, 54.10],
        "ind_AUC":   [97.72, 97.39, 86.68, 78.83, 91.24, 59.99, 61.23, 62.38, 55.86],
    },
    "CAWN": {
        "trans_AP":  [98.76, 99.11, 95.18, 89.56, 80.15, 65.39, 52.84, 70.58, 69.82],
        "trans_AUC": [98.54, 99.01, 93.87, 90.45, 80.38, 68.54, 53.09, 77.16, 75.70],
        "ind_AP":    [98.24, 98.62, 92.73, 86.35, 81.42, 65.24, 49.94, 53.17, 55.80],
        "ind_AUC":   [98.03, 98.42, 90.40, 87.02, 81.86, 67.05, 48.34, 51.49, 58.83],
    },
    "EdgeBank": {
        "trans_AP":  [90.37, 94.86, 76.20, 83.53, 57.97, 60.41, 58.49, 58.39, 64.55],
        "trans_AUC": [90.78, 95.37, 77.30, 87.05, 60.86, 66.75, 62.97, 62.57, 64.14],
        "ind_AP":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # not reported
        "ind_AUC":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # not reported
    },
    "GraphMixer": {
        "trans_AP":  [97.25, 97.31, 93.25, 82.25, 82.78, 62.61, 52.11, 70.74, 77.04],
        "trans_AUC": [96.92, 97.17, 91.81, 84.38, 84.01, 65.52, 52.46, 76.96, 83.17],
        "ind_AP":    [96.65, 95.26, 91.19, 75.88, 81.41, 62.17, 50.68, 50.71, 55.91],
        "ind_AUC":   [96.30, 94.97, 89.30, 76.51, 82.77, 63.48, 50.04, 47.20, 58.32],
    },
    "TCL": {
        "trans_AP":  [96.47, 97.53, 89.57, 79.70, 82.38, 62.21, 51.90, 69.59, 68.67],
        "trans_AUC": [95.84, 97.42, 87.82, 75.74, 83.12, 64.72, 51.88, 76.27, 72.46],
        "ind_AP":    [96.22, 94.09, 87.36, 76.14, 80.60, 62.21, 51.60, 52.59, 54.30],
        "ind_AUC":   [95.57, 93.80, 84.49, 72.33, 81.43, 63.76, 50.51, 50.43, 55.83],
    },
}

model_styles = {
    # --- 原有 5 个模型 ---
    "CASTDyG":    {"color": "b",          "linestyle": "--",                    "marker": "o"},
    "CARoDyG":    {"color": "r",          "linestyle": ":",                     "marker": "^"},
    "DASTDyG":    {"color": "g",          "linestyle": "-.",                    "marker": "s"},
    "DARoDyG":    {"color": "y",          "linestyle": "-",                     "marker": "D"},
    "RepeatMixer":{"color": "m",          "linestyle": (0, (3, 1, 1, 1, 1, 1)),"marker": "*"},
    # --- 新增 9 个模型 ---
    "TIDFormer":  {"color": "c",          "linestyle": (0, (5, 1)),             "marker": "v"},
    "DyGFormer":  {"color": "k",          "linestyle": (0, (1, 1)),             "marker": "<"},
    "CNEN":       {"color": "darkorange", "linestyle": (0, (5, 5)),             "marker": ">"},
    "TGAT":       {"color": "purple",     "linestyle": (0, (3, 5, 1, 5)),       "marker": "p"},
    "TGN":        {"color": "brown",      "linestyle": (0, (1, 5)),             "marker": "h"},
    "CAWN":       {"color": "deeppink",   "linestyle": (0, (5, 1, 1, 1)),       "marker": "H"},
    "EdgeBank":   {"color": "navy",       "linestyle": (0, (3, 3, 1, 3)),       "marker": "P"},
    "GraphMixer": {"color": "teal",       "linestyle": (0, (4, 1, 1, 1, 1, 1)),"marker": "X"},
    "TCL":        {"color": "olive",      "linestyle": (0, (6, 2, 2, 2)),       "marker": "8"},
}

N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

PREFIX = "Q35"
YMIN, YMAX = 40, 100
out_dir = Path(__file__).resolve().parent


def plot_one_radar(metric: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for model_name, data in models.items():
        if model_name == "EdgeBank" and metric in ("ind_AP", "ind_AUC"):
            continue
        values = data[metric] + data[metric][:1]
        style = model_styles[model_name]
        ax.plot(
            angles, values,
            linewidth=2,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=8,
            color=style["color"],
        )
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(YMIN, YMAX)

    pdf_path = out_dir / f"{PREFIX}-{metric}.pdf"
    png_path = out_dir / f"{PREFIX}-{metric}.png"
    plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"已保存: {pdf_path}")
    print(f"已保存: {png_path}")


def save_legend() -> None:
    fig, ax = plt.subplots(figsize=(28, 0.6))
    ax.axis("off")
    legend_elements = [
        Line2D(
            [0], [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=8,
            linewidth=4,
            label=model_name,
        )
        for model_name, style in model_styles.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(model_styles),
        fontsize=14,
        frameon=True,
    )
    pdf_path = out_dir / f"{PREFIX}-radar-legend.pdf"
    png_path = out_dir / f"{PREFIX}-radar-legend.png"
    plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"已保存: {pdf_path}")
    print(f"已保存: {png_path}")


for metric in metrics:
    plot_one_radar(metric)
save_legend()
