import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# =========================
# 1. Data
# =========================
models = [
    "SimpleDyG (w/o Patch)",
    "SimpleDyG",
    "DADyG",
    "CARDyG",
    "NPADyG",
    "SPSEDyG",
    "SPEDyG",
    "SDPEDyG",
    "RWPEDyG",
    "RoPEDyG",
    "STPEDyG",
]

memory_avg = np.array([
    662.231,
    553.219,
    625.336,
    476.076,
    437.720,
    400.061,
    398.917,
    430.830,
    770.768,
    397.272,
    400.471,
])

param_avg = np.array([
    1.919,
    2.102,
    2.098,
    1.996,
    2.136,
    2.102,
    2.102,
    3.268,
    39.744,
    2.102,
    2.102,
])

time_avg = np.array([
    0.214,
    0.227,
    0.372,
    0.311,
    0.257,
    0.124,
    0.236,
    0.228,
    0.476,
    0.403,
    0.242,
])

# =========================
# 2. Colors
# =========================
colors = {
    "SimpleDyG (w/o Patch)": "#1f6ad6",
    "SimpleDyG": "#4fc3d7",
    "DADyG": "#cc33cc",
    "CARDyG": "#8d63d2",
    "NPADyG": "#1b7f2a",
    "SPSEDyG": "#39d87a",
    "SPEDyG": "#5cc46e",
    "SDPEDyG": "#d62728",
    "RWPEDyG": "#d4b517",
    "RoPEDyG": "#4d6edb",
    "STPEDyG": "#e7a35d",
}

# =========================
# 3. Figure setup
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

fig, (ax_left, ax_right) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(13.8, 7.4),
    dpi=200,
    gridspec_kw={"width_ratios": [4.8, 1.2], "wspace": 0.05}
)

fig.patch.set_facecolor("white")
ax_left.set_facecolor("white")
ax_right.set_facecolor("white")

# =========================
# 4. Bubble sizes
# =========================
# s 是面积，不是半径
scale_factor = 18000
bubble_sizes = time_avg * scale_factor

# =========================
# 5. Scatter plot
# =========================
for ax in [ax_left, ax_right]:
    for i, model in enumerate(models):
        ax.scatter(
            param_avg[i],
            memory_avg[i],
            s=bubble_sizes[i],
            color=colors[model],
            alpha=0.38,
            edgecolors="black",
            linewidths=1.1,
            zorder=3,
            clip_on=False
        )

# =========================
# 6. Broken x-axis ranges
# =========================
ax_left.set_xlim(1.74, 3.68)
ax_right.set_xlim(37.4, 41.9)
ax_left.set_ylim(360, 830)

# y-axis ticks
y_ticks = [400, 500, 600, 700, 800]
ax_left.yaxis.set_major_locator(FixedLocator(y_ticks))
ax_left.yaxis.set_major_formatter(FixedFormatter([str(y) for y in y_ticks]))

# x-axis ticks
ax_left.set_xticks([2.0, 2.5, 3.0, 3.5])
ax_right.set_xticks([38, 40])

# =========================
# 7. Clean style
# =========================
for ax in [ax_left, ax_right]:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=15, width=1.4, length=6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

ax_left.spines["right"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.tick_params(axis="y", left=False, labelleft=False)

ax_left.spines["left"].set_linewidth(1.8)
ax_left.spines["bottom"].set_linewidth(1.8)
ax_right.spines["bottom"].set_linewidth(1.8)
ax_right.spines["right"].set_visible(False)

# =========================
# 8. Axis labels
# =========================
fig.supxlabel("Trainable Parameter Size (MB)", fontsize=24, fontweight="bold", y=0.055)
ax_left.set_ylabel("Memory Usage (MB)", fontsize=24, fontweight="bold")

# =========================
# 9. Draw break marks
# =========================
d = 0.012

# 只保留下边的断轴标记，去掉上边的断轴标记
kwargs = dict(transform=ax_left.transAxes, color="black", clip_on=False, linewidth=1.5)
ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs = dict(transform=ax_right.transAxes, color="black", clip_on=False, linewidth=1.5)
ax_right.plot((-d, +d), (-d, +d), **kwargs)

# =========================
# 10. Label font sizes
# =========================
label_fontsize_left = 14
label_fontsize_right = 15

# =========================
# 11. Standard label positions on left panel
# =========================
# 这里先处理其余模型；RoPEDyG 和 SPSEDyG 单独在后面处理
label_positions_left = {
    "SimpleDyG (w/o Patch)": (1.82, 715),
    "DADyG": (2.33, 645),
    "SimpleDyG": (2.35, 565),
    "CARDyG": (2.42, 505),
    "NPADyG": (2.42, 465),
    "SPEDyG": (2.42, 430),
    "STPEDyG": (2.47, 394),
    "SDPEDyG": (3.40, 465),
}

align_left = {
    "SimpleDyG (w/o Patch)": ("left", "center"),
    "DADyG": ("left", "center"),
    "SimpleDyG": ("left", "center"),
    "CARDyG": ("left", "center"),
    "NPADyG": ("left", "center"),
    "SPEDyG": ("left", "center"),
    "STPEDyG": ("left", "center"),
    "SDPEDyG": ("left", "center"),
}

for model, (tx, ty) in label_positions_left.items():
    i = models.index(model)
    ha, va = align_left[model]

    ax_left.annotate(
        f"{model}\n{time_avg[i]:.3f} s",
        xy=(param_avg[i], memory_avg[i]),
        xytext=(tx, ty),
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=label_fontsize_left,
        fontweight="bold",
        color=colors[model],
        arrowprops=dict(
            arrowstyle="-",
            color=colors[model],
            lw=1.0,
            alpha=0.9,
            shrinkA=0,
            shrinkB=5
        ),
        zorder=5,
        clip_on=False
    )

# =========================
# 12. Special labels: RoPEDyG and SPSEDyG
# =========================
# 单独放到左下角空白区域，不与圆圈重叠
# 使用 offset points 更容易稳定控制位置

# RoPEDyG
i = models.index("RoPEDyG")
ax_left.annotate(
    f"RoPEDyG\n{time_avg[i]:.3f} s",
    xy=(param_avg[i], memory_avg[i]),
    xytext=(-50, 20),                 # 往左上挪到左下空白区域
    textcoords="offset points",
    ha="right",
    va="center",
    fontsize=label_fontsize_left,
    fontweight="bold",
    color=colors["RoPEDyG"],
    arrowprops=dict(
        arrowstyle="-",
        color=colors["RoPEDyG"],
        lw=1.0,
        alpha=0.9,
        shrinkA=0,
        shrinkB=5
    ),
    zorder=5,
    clip_on=False
)

# SPSEDyG
i = models.index("SPSEDyG")
ax_left.annotate(
    f"SPSEDyG\n{time_avg[i]:.3f} s",
    xy=(param_avg[i], memory_avg[i]),
    xytext=(-48, -20),                # 往左下挪到空白区域
    textcoords="offset points",
    ha="right",
    va="center",
    fontsize=label_fontsize_left,
    fontweight="bold",
    color=colors["SPSEDyG"],
    arrowprops=dict(
        arrowstyle="-",
        color=colors["SPSEDyG"],
        lw=1.0,
        alpha=0.9,
        shrinkA=0,
        shrinkB=5
    ),
    zorder=5,
    clip_on=False
)

# =========================
# 13. Label on right panel
# =========================
i = models.index("RWPEDyG")
ax_right.annotate(
    f"RWPEDyG\n{time_avg[i]:.3f} s",
    xy=(param_avg[i], memory_avg[i]),
    xytext=(40.25, 802),
    textcoords="data",
    ha="left",
    va="center",
    fontsize=label_fontsize_right,
    fontweight="bold",
    color=colors["RWPEDyG"],
    arrowprops=dict(
        arrowstyle="-",
        color=colors["RWPEDyG"],
        lw=1.0,
        alpha=0.9,
        shrinkA=0,
        shrinkB=5
    ),
    zorder=5,
    clip_on=False
)

# =========================
# 14. Layout and save
# =========================
plt.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.98, wspace=0.05)

plt.savefig("lp_efficiency_bubble_chart_broken_x_final.pdf", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_broken_x_final.svg", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_broken_x_final.eps", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_broken_x_final.png", dpi=300, bbox_inches="tight")

plt.show()