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
        "SPEDyG":  {"AUC": [87.14, 68.94]},
        "SDPEDyG": {"AUC": [87.15, 69.26]},
        "RWPEDyG": {"AUC": [87.02, 69.11]},
        "RoPEDyG": {"AUC": [87.02, 69.85]},
        "STPEDyG": {"AUC": [87.93, 69.43]},
    },
    "backbone_classifier": {
        "SPEDyG":  {"AUC": [86.64, 72.66]},
        "SDPEDyG": {"AUC": [86.75, 73.98]},
        "RWPEDyG": {"AUC": [86.54, 73.07]},
        "RoPEDyG": {"AUC": [86.98, 72.60]},
        "STPEDyG": {"AUC": [88.27, 73.09]},
    },
}

setting_titles = {
    "only_classifier": "Only Classifier",
    "backbone_classifier": "Backbone + Classifier",
}

output_basenames = {
    "only_classifier": "wiki_reddit_auc_q25_only_classifier",
    "backbone_classifier": "wiki_reddit_auc_q25_backbone_classifier",
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
