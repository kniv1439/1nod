# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18

# AUC-ROC on Wikipedia and Reddit: [Wikipedia, Reddit]
# 请按实验结果修改下列数值
models_by_setting = {
    "only_classifier": {
        "SimpleDyG": {"AUC": [86.76, 68.83]},
        "DADyG":     {"AUC": [85.35, 65.67]},
        "CARDyG":    {"AUC": [84.15, 64.16]},
        "NPADyG":    {"AUC": [86.07, 65.34]},
        "SPSEDyG":   {"AUC": [83.64, 66.18]},
    },
    "backbone_classifier": {
        "SimpleDyG": {"AUC": [86.45, 72.55]},
        "DADyG":     {"AUC": [85.79, 71.44]},
        "CARDyG":    {"AUC": [86.05, 72.47]},
        "NPADyG":    {"AUC": [86.39, 72.50]},
        "SPSEDyG":   {"AUC": [85.32, 70.23]},
    },
}

setting_titles = {
    "only_classifier": "Only Classifier",
    "backbone_classifier": "Backbone + Classifier",
}

output_basenames = {
    "only_classifier": "wiki_reddit_auc_only_classifier",
    "backbone_classifier": "wiki_reddit_auc_backbone_classifier",
}

dataset_colors = {
    "Wikipedia": "steelblue",
    "Reddit": "coral",
}

bar_width = 0.35


def plot_setting(setting_key, models, out_dir):
    model_names = list(models.keys())
    n = len(model_names)
    x = np.arange(n)

    wiki_heights = [models[name]["AUC"][0] for name in model_names]
    reddit_heights = [models[name]["AUC"][1] for name in model_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(setting_titles[setting_key], fontsize=22, y=1.02)

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

    fig.subplots_adjust(bottom=0.28, top=0.88)

    base = output_basenames[setting_key]
    pdf_path = os.path.join(out_dir, f"{base}.pdf")
    png_path = os.path.join(out_dir, f"{base}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return pdf_path, png_path


out_dir = os.path.dirname(os.path.abspath(__file__))
print("已保存:")
for setting_key, models in models_by_setting.items():
    pdf_path, png_path = plot_setting(setting_key, models, out_dir)
    print(f"  [{setting_titles[setting_key]}]")
    print(f"    {pdf_path}")
    print(f"    {png_path}")
