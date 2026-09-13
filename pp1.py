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
    0.275,
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

fig, ax = plt.subplots(figsize=(13.2, 7.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# =========================
# 4. Bubble sizes
# =========================
# 圆圈面积与 param_avg 成正比
bubble_scale = 650
bubble_sizes = param_avg * bubble_scale

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
# 进一步扩大范围，保证右上角 RWPEDyG 大圆完整显示
ax.set_xlim(0.09, 0.70)
ax.set_ylim(330, 1000)

x_ticks = np.arange(0.10, 0.56, 0.05)
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{x:.2f}" for x in x_ticks]))

y_ticks = [400, 500, 600, 700, 800, 900]
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
# 这里用数据坐标直接控制标签位置
label_positions = {
    "SimpleDyG (w/o Patch)": (0.220, 715),
    "SimpleDyG": (0.248, 585),
    "DADyG": (0.362, 685),
    "CARDyG": (0.305, 535),
    "NPADyG": (0.256, 490),
    "SPSEDyG": (0.162, 448),
    "SPEDyG": (0.174, 365),
    "SDPEDyG": (0.170, 480),
    "RWPEDyG": (0.550, 815),
    "RoPEDyG": (0.382, 420),
    "STPEDyG": (0.275, 378),
}

alignments = {
    "SimpleDyG (w/o Patch)": ("left", "center"),
    "SimpleDyG": ("left", "center"),
    "DADyG": ("left", "center"),
    "CARDyG": ("center", "center"),
    "NPADyG": ("center", "center"),
    "SPSEDyG": ("right", "center"),
    "SPEDyG": ("left", "center"),
    "SDPEDyG": ("left", "center"),
    "RWPEDyG": ("left", "center"),
    "RoPEDyG": ("right", "center"),
    "STPEDyG": ("left", "center"),
}

# =========================
# 11. Annotate labels
# =========================
# 第二行显示参数规模，因为气泡大小对应 param_avg
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

plt.savefig("lp_efficiency_bubble_chart_time_memory_paramsize_final.pdf", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_time_memory_paramsize_final.svg", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_time_memory_paramsize_final.eps", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_time_memory_paramsize_final.png", dpi=300, bbox_inches="tight")

plt.show()