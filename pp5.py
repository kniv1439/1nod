import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# =========================
# 1. Data: Node Classification / Only Classifier / Remaining Models
# =========================
models = [
    'SimpleDyG (w/o Patch)',
    'SimpleDyG',
    'DADyG',
    'CARDyG',
    'NPADyG',
    'SPSEDyG',
    'SPEDyG',
    'SDPEDyG',
    'RWPEDyG',
    'RoPEDyG',
    'STPEDyG',
]

memory_avg = np.array([
    403.969,
    354.309,
    354.310,
    376.459,
    354.412,
    354.310,
    354.441,
    362.009,
    456.896,
    358.874,
    354.951,
])

param_avg = np.array([
    1.800,
    1.983,
    1.980,
    1.878,
    2.017,
    1.983,
    1.983,
    3.149,
    39.625,
    1.983,
    1.983,
])

time_avg = np.array([
    0.115,
    0.115,
    0.206,
    0.187,
    0.156,
    0.176,
    0.142,
    0.117,
    0.228,
    0.377,
    0.139,
])

# =========================
# 2. Colors
# =========================
colors = {
    'SimpleDyG (w/o Patch)': '#1f6ad6',
    'SimpleDyG': '#4fc3d7',
    'DADyG': '#cc33cc',
    'CARDyG': '#8d63d2',
    'NPADyG': '#1b7f2a',
    'SPSEDyG': '#39d87a',
    'SPEDyG': '#5cc46e',
    'SDPEDyG': '#d62728',
    'RWPEDyG': '#d4b517',
    'RoPEDyG': '#4d6edb',
    'STPEDyG': '#e7a35d',
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
ax.set_xlim(0.05, 0.42)
ax.set_ylim(320, 540)

x_ticks = np.array([0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42])
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{x:.2f}" for x in x_ticks]))

y_ticks = [350, 400, 450, 500]
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
    'SimpleDyG (w/o Patch)': (0.105, 425.000),
    'SimpleDyG': (0.130, 335.000),
    'DADyG': (0.200, 335.000),
    'CARDyG': (0.180, 394.000),
    'NPADyG': (0.166, 350.000),
    'SPSEDyG': (0.090, 370.000),
    'SPEDyG': (0.135, 371.000),
    'SDPEDyG': (0.102, 383.000),
    'RWPEDyG': (0.255, 488.000),
    'RoPEDyG': (0.360, 375.000),
    'STPEDyG': (0.134, 335.000),
}

alignments = {
    'SimpleDyG (w/o Patch)': ('left', 'center'),
    'SimpleDyG': ('right', 'center'),
    'DADyG': ('left', 'center'),
    'CARDyG': ('left', 'center'),
    'NPADyG': ('left', 'center'),
    'SPSEDyG': ('right', 'center'),
    'SPEDyG': ('left', 'center'),
    'SDPEDyG': ('left', 'center'),
    'RWPEDyG': ('left', 'center'),
    'RoPEDyG': ('left', 'center'),
    'STPEDyG': ('left', 'center'),
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

plt.savefig("nc_only_classifier_variants_bubble.pdf", bbox_inches="tight")
plt.savefig("nc_only_classifier_variants_bubble.svg", bbox_inches="tight")
plt.savefig("nc_only_classifier_variants_bubble.png", dpi=300, bbox_inches="tight")

plt.show()