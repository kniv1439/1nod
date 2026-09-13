# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无图形界面时也能保存文件
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18

# AUC-ROC on Wikipedia and Reddit (from transductive results)
models = {
    "CASTDyG":     {"AUC": [87.04, 66.68]},
    "CARoDyG":     {"AUC": [86.65, 66.64]},
    "DASTDyG":     {"AUC": [86.97, 66.25]},
    "DARoDyG":     {"AUC": [85.43, 65.97]},
    "RepeatMixer": {"AUC": [80.94, 68.12]},
    "TIDFormer":   {"AUC": [87.53, 69.59]},
    "DyGFormer":   {"AUC": [86.76, 68.83]},
    "CNEN":        {"AUC": [88.37, 66.84]},
    "TGAT":        {"AUC": [84.09, 70.04]},
    "TGN":         {"AUC": [86.38, 63.27]},
    "CAWN":        {"AUC": [84.88, 66.34]},
    "GraphMixer":  {"AUC": [86.80, 64.22]},
    "TCL":         {"AUC": [77.83, 68.87]},
}

dataset_colors = {
    "Wikipedia": "steelblue",
    "Reddit": "coral",
}

model_names = list(models.keys())
n = len(model_names)
x = np.arange(n)
bar_width = 0.35

wiki_heights = [models[name]["AUC"][0] for name in model_names]
reddit_heights = [models[name]["AUC"][1] for name in model_names]

fig, ax = plt.subplots(figsize=(24, 7))

ax.bar(
    x - bar_width / 2,
    wiki_heights,
    bar_width,
    label="Wikipedia",
    color=dataset_colors["Wikipedia"],
)
ax.bar(
    x + bar_width / 2,
    reddit_heights,
    bar_width,
    label="Reddit",
    color=dataset_colors["Reddit"],
)
ax.set_xticks(x)
ax.set_xticklabels(
    model_names,
    rotation=45,
    ha="right",
    fontsize=12,
)
ax.set_ylabel("AUC-ROC (%)")
ax.set_ylim(60, 90)
ax.legend(loc="upper right", fontsize=14)
ax.tick_params(axis="x", pad=2)

fig.subplots_adjust(bottom=0.28)
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "wiki_reddit_auc_bar.pdf")
png_path = os.path.join(out_dir, "wiki_reddit_auc_bar.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight", dpi=150)
plt.close()
print("已保存:")
print(f"  {pdf_path}")
print(f"  {png_path}")
