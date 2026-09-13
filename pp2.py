import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# =========================
# 1. Data: Node Classification / Backbone & Classifier / Baselines
# =========================
models = [
    'RepeatMixer',
    'TIDFormer',
    'DyGFormer',
    'CNEN',
    'TGAT',
    'TGN',
    'CAWN',
    'GraphMixer',
    'TCL',
    'SDPEDyG',
    'STPEDyG',
    'NPASDPEDyG',
    'NPASTDyG',
]

memory_avg = np.array([
    641.435,
    1006.317,
    680.787,
    378.243,
    1748.490,
    366.837,
    1790.189,
    1206.995,
    898.737,
    458.187,
    434.515,
    495.200,
    463.379,
])

param_avg = np.array([
    7.096,
    4.024,
    4.218,
    0.613,
    4.032,
    3.677,
    15.912,
    2.391,
    3.358,
    3.149,
    1.983,
    3.184,
    2.017,
])

time_avg = np.array([
    0.099,
    0.153,
    0.196,
    0.080,
    0.182,
    0.108,
    0.283,
    0.087,
    0.116,
    0.155,
    0.167,
    0.138,
    0.173,
])

# =========================
# 2. Colors
# =========================
colors = {
    'RepeatMixer': '#9467bd',
    'TIDFormer': '#8c564b',
    'DyGFormer': '#1f77b4',
    'CNEN': '#2ca02c',
    'TGAT': '#ff7f0e',
    'TGN': '#17becf',
    'CAWN': '#7f7f7f',
    'GraphMixer': '#bcbd22',
    'TCL': '#e377c2',
    'SDPEDyG': '#d62728',
    'STPEDyG': '#e7a35d',
    'NPASDPEDyG': '#b45309',
    'NPASTDyG': '#008c72',
}

# =========================
# 3. Figure setup
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

fig, ax = plt.subplots(figsize=(13.2, 7.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# =========================
# 4. Bubble sizes
# =========================
# 圆圈面积与 param_avg 成正比。
bubble_scale = 650
min_visible_size = 70
bubble_sizes = np.maximum(param_avg * bubble_scale, min_visible_size)

# =========================
# 5. Scatter plot
# =========================
for i, model in enumerate(models):
    ax.scatter(
        time_avg[i],
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
# 6. Axis limits and ticks
# =========================
ax.set_xlim(0.06, 0.32)
ax.set_ylim(250, 2050)

x_ticks = np.array([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{x:.2f}" for x in x_ticks]))

y_ticks = [400, 800, 1200, 1600, 2000]
ax.yaxis.set_major_locator(FixedLocator(y_ticks))
ax.yaxis.set_major_formatter(FixedFormatter([str(y) for y in y_ticks]))

# =========================
# 7. Clean style
# =========================
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.8)
ax.spines["bottom"].set_linewidth(1.8)

ax.tick_params(axis="both", which="major", labelsize=18, width=1.4, length=6)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("bold")

# =========================
# 8. Axis labels
# =========================
ax.set_xlabel("Training Time of the First Step (sec)", fontsize=27, fontweight="bold")
ax.set_ylabel("Memory Usage (MB)", fontsize=27, fontweight="bold")

# =========================
# 9. Label font size
# =========================
label_fontsize = 16

# =========================
# 10. Manual label positions
# =========================
label_positions = {
    'RepeatMixer': (0.064, 750.000),
    'TIDFormer': (0.160, 1112.000),
    'DyGFormer': (0.205, 620.000),
    'CNEN': (0.065, 472.000),
    'TGAT': (0.192, 1660.000),
    'TGN': (0.120, 330.000),
    'CAWN': (0.300, 1900.000),
    'GraphMixer': (0.094, 1300.000),
    'TCL': (0.125, 950.000),
    'SDPEDyG': (0.143, 660.000),
    'STPEDyG': (0.180, 396.000),
    'NPASDPEDyG': (0.112, 700.000),
    'NPASTDyG': (0.162, 587.000),
}

alignments = {
    'RepeatMixer': ('left', 'center'),
    'TIDFormer': ('left', 'center'),
    'DyGFormer': ('left', 'center'),
    'CNEN': ('left', 'center'),
    'TGAT': ('left', 'center'),
    'TGN': ('left', 'center'),
    'CAWN': ('left', 'center'),
    'GraphMixer': ('left', 'center'),
    'TCL': ('left', 'center'),
    'SDPEDyG': ('left', 'center'),
    'STPEDyG': ('left', 'center'),
    'NPASDPEDyG': ('left', 'center'),
    'NPASTDyG': ('left', 'center'),
}

# =========================
# 11. Annotate labels
# =========================
# 第二行显示参数规模，因为气泡大小对应 param_avg。
for i, model in enumerate(models):
    tx, ty = label_positions[model]
    ha, va = alignments[model]

    ax.annotate(
        f"{model}\n{param_avg[i]:.1f} MB",
        xy=(time_avg[i], memory_avg[i]),
        xytext=(tx, ty),
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=label_fontsize,
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
# 12. Layout and save
# =========================
plt.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.97)

plt.savefig("nc_backbone_classifier_baselines_bubble.pdf", bbox_inches="tight")
plt.savefig("nc_backbone_classifier_baselines_bubble.svg", bbox_inches="tight")
plt.savefig("nc_backbone_classifier_baselines_bubble.png", dpi=300, bbox_inches="tight")

plt.show()