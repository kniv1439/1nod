import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# =========================
# 1. Data
# =========================
models = [
    "CASTDyG",
    "CARoDyG",
    "DASTDyG",
    "DARoDyG",
    "RepeatMixer",
    "TIDFormer",
    "DyGFormer",
    "CNEN",
    "TGAT",
    "TGN",
    "CAWN",
    "EdgeBank",
    "GraphMixer",
    "TCL",
]

memory_avg = np.array([
    478.375,
    478.415,
    627.166,
    625.825,
    1497.357,
    1114.394,
    620.603,
    308.610,
    2250.957,
    322.255,
    3920.480,
    0.000,
    1260.860,
    1310.968,
])

param_avg = np.array([
    1.996,
    1.996,
    2.098,
    2.098,
    7.274,
    4.209,
    6.102,
    0.793,
    4.212,
    3.856,
    16.091,
    0.000,
    2.570,
    3.538,
])

time_avg = np.array([
    0.352,
    0.543,
    0.386,
    0.589,
    0.125,
    0.247,
    0.307,
    0.100,
    0.270,
    0.120,
    0.675,
    0.084,
    0.127,
    0.158,
])

# =========================
# 2. Colors
# =========================
colors = {
    "CASTDyG": "#9aa7e6",
    "CARoDyG": "#f08a00",
    "DASTDyG": "#d7df72",
    "DARoDyG": "#f06b6b",
    "RepeatMixer": "#9467bd",
    "TIDFormer": "#8c564b",
    "DyGFormer": "#1f77b4",
    "CNEN": "#2ca02c",
    "TGAT": "#ff7f0e",
    "TGN": "#17becf",
    "CAWN": "#7f7f7f",
    "EdgeBank": "#000000",
    "GraphMixer": "#bcbd22",
    "TCL": "#e377c2",
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
# EdgeBank 的参数规模为 0；为了在图中能看到它，这里给一个很小的可视化下限。
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
ax.set_xlim(0.05, 0.76)
ax.set_ylim(-250, 4400)

x_ticks = np.arange(0.10, 0.76, 0.10)
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{x:.2f}" for x in x_ticks]))

y_ticks = [0, 1000, 2000, 3000, 4000]
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

ax.tick_params(axis="both", which="major", labelsize=15, width=1.4, length=6)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("bold")

# =========================
# 8. Axis labels
# =========================
ax.set_xlabel("Training Time of the First Step (s)", fontsize=24, fontweight="bold")
ax.set_ylabel("Memory Usage (MB)", fontsize=24, fontweight="bold")

# =========================
# 9. Label font size
# =========================
label_fontsize = 13

# =========================
# 10. Manual label positions
# =========================
label_positions = {
    "CASTDyG": (0.330, 934),
    "CARoDyG": (0.530, 815),
    "DASTDyG": (0.405, 820),
    "DARoDyG": (0.615, 780),
    "RepeatMixer": (0.105, 2020),
    "TIDFormer": (0.235, 1525),
    "DyGFormer": (0.300, 112),
    "CNEN": (0.060, 620),
    "TGAT": (0.295, 2350),
    "TGN": (0.147, 190),
    "CAWN": (0.570, 4100),
    "EdgeBank": (0.088, -95),
    "GraphMixer": (0.100, 825),
    "TCL": (0.185, 1450),
}

alignments = {
    "CASTDyG": ("left", "center"),
    "CARoDyG": ("left", "center"),
    "DASTDyG": ("left", "center"),
    "DARoDyG": ("left", "center"),
    "RepeatMixer": ("left", "center"),
    "TIDFormer": ("left", "center"),
    "DyGFormer": ("left", "center"),
    "CNEN": ("left", "center"),
    "TGAT": ("left", "center"),
    "TGN": ("left", "center"),
    "CAWN": ("left", "center"),
    "EdgeBank": ("left", "center"),
    "GraphMixer": ("left", "center"),
    "TCL": ("left", "center"),
}

# =========================
# 11. Annotate labels
# =========================
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

plt.savefig("lp_efficiency_bubble_chart_baselines_time_memory_paramsize.pdf", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_baselines_time_memory_paramsize.svg", bbox_inches="tight")
plt.savefig("lp_efficiency_bubble_chart_baselines_time_memory_paramsize.png", dpi=300, bbox_inches="tight")

plt.show()
