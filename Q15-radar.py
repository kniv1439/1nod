#!/usr/bin/env python3
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
    # data order matches labels: Wikipedia, Reddit, Uci, Enron, Mooc, UNtrade, UNvote, USLegis, CanParl
    "SimpleDyG": {
        "trans_AP":  [98.60, 99.00, 95.44, 91.69, 85.35, 63.74, 55.80, 69.43, 89.78],
        "trans_AUC": [98.45, 98.87, 94.27, 92.71, 85.52, 67.10, 56.06, 76.38, 92.37],
        "ind_AP":    [98.09, 98.52, 94.31, 89.68, 85.23, 62.86, 55.38, 50.11, 72.14],
        "ind_AUC":   [97.90, 98.29, 91.15, 90.24, 85.27, 64.96, 52.94, 47.57, 75.66],
    },
    "DADyG": {
        "trans_AP":  [98.85, 99.28, 96.24, 94.01, 95.35, 92.88, 83.67, 83.04, 97.39],
        "trans_AUC": [98.86, 99.04, 96.05, 93.72, 94.98, 91.87, 81.59, 77.61, 97.52],
        "ind_AP":    [98.49, 99.01, 94.93, 90.92, 91.62, 89.58, 64.03, 64.70, 95.43],
        "ind_AUC":   [98.48, 98.64, 95.18, 90.51, 90.34, 89.63, 62.58, 55.78, 95.67],
    },
    "CARDyG": {
        "trans_AP":  [99.05, 99.26, 95.98, 93.12, 84.68, 67.86, 56.65, 73.22, 93.58],
        "trans_AUC": [98.91, 99.17, 95.06, 92.98, 84.77, 70.68, 56.42, 77.27, 95.69],
        "ind_AP":    [98.27, 98.84, 94.82, 90.48, 84.11, 67.26, 55.87, 57.20, 89.72],
        "ind_AUC":   [98.07, 98.68, 91.15, 90.36, 84.41, 68.78, 54.06, 53.40, 92.25],
    },
    "NPADyG": {
        "trans_AP":  [98.58, 98.99, 96.20, 91.53, 85.32, 64.20, 50.91, 69.59, 91.65],
        "trans_AUC": [98.44, 98.86, 95.41, 92.56, 85.50, 67.43, 51.37, 76.45, 93.62],
        "ind_AP":    [98.09, 98.49, 94.50, 89.81, 84.53, 63.18, 48.74, 51.67, 86.50],
        "ind_AUC":   [97.89, 98.25, 90.86, 90.06, 85.23, 65.29, 47.88, 48.79, 89.20],
    },
    "SPSEDyG": {
        "trans_AP":  [98.61, 99.02, 95.53, 91.76, 86.01, 89.08, 87.04, 80.42, 90.03],
        "trans_AUC": [98.47, 98.91, 94.63, 92.74, 86.73, 86.87, 84.70, 81.19, 92.94],
        "ind_AP":    [98.15, 98.52, 94.39, 89.70, 85.93, 88.09, 86.98, 67.18, 73.06],
        "ind_AUC":   [97.92, 98.30, 91.85, 90.32, 86.33, 85.04, 84.43, 58.72, 75.80],
    },
}

model_styles = {
    "SimpleDyG": {"color": "b", "linestyle": "--",             "marker": "o"},
    "DADyG":     {"color": "r", "linestyle": ":",              "marker": "^"},
    "CARDyG":    {"color": "g", "linestyle": "-.",             "marker": "s"},
    "NPADyG":    {"color": "y", "linestyle": "-",              "marker": "D"},
    "SPSEDyG":   {"color": "m", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "*"},
}

N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

PREFIX = "Q15"
YMIN, YMAX = 40, 100
out_dir = Path(__file__).resolve().parent


def plot_one_radar(metric: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for model_name, data in models.items():
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
    fig, ax = plt.subplots(figsize=(12, 0.6))
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
