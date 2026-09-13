#!/usr/bin/env python3
import os
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
    # data order matches labels: Wikipedia, Reddit, Uci, Enron, Mooc, UNtrade, UNvote, USLegis, CanParl
    "SPEDyG": {
        "trans_AP":  [98.87, 99.24, 95.60, 91.80, 86.37, 66.09, 56.06, 70.42, 94.60],
        "trans_AUC": [98.74, 99.18, 94.48, 92.78, 87.17, 68.14, 56.61, 77.73, 95.89],
        "ind_AP":    [98.31, 98.85, 94.91, 89.70, 85.64, 64.68, 55.82, 54.22, 89.72],
        "ind_AUC":   [98.16, 98.73, 92.26, 90.39, 86.80, 65.88, 53.42, 53.78, 92.11],
    },
    "SDPEDyG": {
        "trans_AP":  [98.89, 99.06, 95.91, 91.92, 86.09, 64.62, 56.23, 72.58, 91.45],
        "trans_AUC": [98.70, 98.89, 94.92, 92.80, 85.69, 67.23, 57.16, 77.67, 93.40],
        "ind_AP":    [98.16, 98.63, 94.37, 89.77, 85.60, 63.18, 55.59, 55.36, 85.29],
        "ind_AUC":   [97.92, 98.33, 91.73, 90.43, 85.68, 65.87, 53.35, 51.59, 87.99],
    },
    "RWPEDyG": {
        "trans_AP":  [98.96, 99.19, 96.31, 92.18, 86.50, 68.67, 56.03, 70.52, 91.96],
        "trans_AUC": [98.81, 99.08, 95.47, 92.89, 85.79, 71.46, 56.42, 76.53, 94.35],
        "ind_AP":    [98.15, 98.70, 94.80, 89.83, 86.36, 67.52, 55.51, 54.09, 85.97],
        "ind_AUC":   [97.93, 98.47, 92.03, 90.36, 86.00, 69.11, 53.38, 51.30, 89.28],
    },
    "RoPEDyG": {
        "trans_AP":  [98.96, 99.09, 96.83, 92.61, 88.40, 65.33, 56.88, 72.60, 95.36],
        "trans_AUC": [98.81, 98.92, 96.08, 93.09, 88.57, 69.09, 58.72, 77.94, 96.45],
        "ind_AP":    [98.23, 98.64, 95.27, 89.96, 87.85, 65.18, 56.47, 55.68, 90.54],
        "ind_AUC":   [97.97, 98.38, 92.71, 90.44, 87.42, 67.16, 55.24, 54.97, 92.78],
    },
    "STPEDyG": {
        "trans_AP":  [99.14, 99.19, 97.00, 92.09, 87.17, 66.51, 57.58, 72.81, 93.70],
        "trans_AUC": [99.05, 99.09, 96.40, 92.82, 87.70, 70.22, 59.16, 79.35, 95.20],
        "ind_AP":    [98.56, 98.75, 95.78, 89.75, 86.72, 65.23, 56.68, 55.69, 87.19],
        "ind_AUC":   [98.45, 98.51, 93.08, 90.38, 87.67, 67.79, 56.81, 55.96, 89.46],
    },
}

model_styles = {
    "SPEDyG":   {"color": "b", "linestyle": "--",             "marker": "o"},
    "SDPEDyG":  {"color": "r", "linestyle": ":",              "marker": "^"},
    "RWPEDyG":  {"color": "g", "linestyle": "-.",             "marker": "s"},
    "RoPEDyG":  {"color": "y", "linestyle": "-",              "marker": "D"},
    "STPEDyG":  {"color": "m", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "*"},
}

N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 4, figsize=(24, 6), subplot_kw=dict(polar=True))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for model_name, data in models.items():
        values = data[metric] + data[metric][:1]
        style = model_styles[model_name]
        ax.plot(
            angles, values,
            linewidth=2,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=8,
            label=model_name,
            color=style["color"],
        )
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(50, 100)
    ax.set_title(metric_titles[metric], size=18, pad=12)

legend_elements = [
    Line2D(
        [0], [0],
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=8,
        linewidth=2,
        label=model_name,
    )
    for model_name, style in model_styles.items()
]
fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.06),
    ncol=5,
    fontsize=14,
)

plt.tight_layout(rect=[0, 0, 1, 0.90])

out_dir = Path(__file__).resolve().parent
pdf_path = out_dir / "Q25-radar.pdf"
png_path = out_dir / "Q25-radar.png"
plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
plt.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"已保存: {pdf_path}")
print(f"已保存: {png_path}")

if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    plt.show()
else:
    plt.close()
    print("当前无图形界面，已跳过窗口显示；请用 PDF/PNG 查看结果。")
