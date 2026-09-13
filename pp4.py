import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# =========================
# 1. Data: Node Classification / Only Classifier / Baselines
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
    404.526,
    425.809,
    427.725,
    352.198,
    788.904,
    350.770,
    810.266,
    972.436,
    371.592,
    362.009,
    354.951,
    365.141,
    356.811,
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
    0.084,
    0.144,
    0.160,
    0.067,
    0.186,
    0.109,
    0.277,
    0.091,
    0.100,
    0.117,
    0.139,
    0.246,
    0.292,
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
ax.set_xlim(0.05, 0.32)
ax.set_ylim(300, 1100)

x_ticks = np.array([0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30])
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{x:.2f}" for x in x_ticks]))

y_ticks = [400, 600, 800, 1000]
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
    'RepeatMixer': (0.060, 508.000),
    'TIDFormer': (0.130, 505.000),
    'DyGFormer': (0.170, 435.000),
    'CNEN': (0.052, 395.000),
    'TGAT': (0.198, 760.000),
    'TGN': (0.102, 509.000),
    'CAWN': (0.290, 890.000),
    'GraphMixer': (0.086, 1035.000),
    'TCL': (0.090, 480.000),
    'SDPEDyG': (0.112, 441.000),
    'STPEDyG': (0.149, 330.000),
    'NPASDPEDyG': (0.239, 432.000),
    'NPASTDyG': (0.280, 419.000),
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

plt.savefig("nc_only_classifier_baselines_bubble.pdf", bbox_inches="tight")
plt.savefig("nc_only_classifier_baselines_bubble.svg", bbox_inches="tight")
plt.savefig("nc_only_classifier_baselines_bubble.png", dpi=300, bbox_inches="tight")

plt.show()