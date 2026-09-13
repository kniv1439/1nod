# -*- coding: utf-8 -*-
"""
四数据集过平滑趋势分析：随 patch size 与二阶邻居采样数(hop) 增大的变化。
仅读取 oversmoothing_final_layer.json，不加载模型。
指标含义：cos_sim_mean ↑ = 越过平滑；l2_dist_mean ↓ = 越过平滑。
"""

import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 避免中文在图中乱码：标题用英文，总结用 trend_summary.txt
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, 'oversmoothing_final_layer.json')
OUT_DIR = SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# 四数据集
DATASETS = ['CanParl', 'USLegis', 'uci', 'wikipedia']
DATASET_LABELS = {'CanParl': 'CanParl', 'USLegis': 'USLegis', 'uci': 'UCI', 'wikipedia': 'Wikipedia'}

# 配色
PATCH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
PATCH_MARKERS = ['o', 's', '^', 'D', 'v', 'P']
HOP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
HOP_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']


def parse_key(key):
    """从 key 解析 dataset, seed, hop, patch。key 格式: {dataset}_seed{s}_hop{h}_patch{p}"""
    m = re.match(r'^(.+?)_seed(\d+)_hop(\d+)_patch(\d+)$', key)
    if not m:
        return None
    return {
        'dataset': m.group(1),
        'seed': int(m.group(2)),
        'hop': int(m.group(3)),
        'patch': int(m.group(4)),
    }


def load_records():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    records = []
    for key, metrics in raw.items():
        p = parse_key(key)
        if p is None:
            continue
        r = p.copy()
        r.update(metrics)
        records.append(r)
    return records


def filter_four_datasets(records):
    return [r for r in records if r['dataset'] in DATASETS]


# ─── 1. 随 hop（二阶邻居采样数）增大：每条线一个 patch ─────────────────────────
def plot_trend_vs_hop(records_by_ds, metric, ylabel, title_suffix, filename):
    """x=hop, 每条线=一个 patch size。cos_sim 越高越过平滑，l2 越低越过平滑。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        ds_rec = records_by_ds.get(ds, [])
        patches = sorted(set(r['patch'] for r in ds_rec))
        for ci, patch in enumerate(patches):
            pts = sorted([r for r in ds_rec if r['patch'] == patch], key=lambda x: x['hop'])
            if len(pts) < 2:
                continue
            hops = [r['hop'] for r in pts]
            vals = [r[metric] for r in pts]
            ax.plot(hops, vals,
                    marker=PATCH_MARKERS[ci % len(PATCH_MARKERS)],
                    color=PATCH_COLORS[ci % len(PATCH_COLORS)],
                    linewidth=2, markersize=7, label=f'patch={patch}')
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.set_xlabel('Max 2nd-hop (num. 2nd-order neighbors)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Oversmoothing vs. 2nd-hop size — ' + title_suffix, fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {filename}')


# ─── 2. 随 patch size 增大：每条线一个 hop ─────────────────────────────────────
def plot_trend_vs_patch(records_by_ds, metric, ylabel, title_suffix, filename):
    """x=patch, 每条线=一个 hop。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        ds_rec = records_by_ds.get(ds, [])
        hops = sorted(set(r['hop'] for r in ds_rec))
        for ci, hop in enumerate(hops):
            pts = sorted([r for r in ds_rec if r['hop'] == hop], key=lambda x: x['patch'])
            if len(pts) < 2:
                continue
            patches = [r['patch'] for r in pts]
            vals = [r[metric] for r in pts]
            ax.plot(patches, vals,
                    marker=HOP_MARKERS[ci % len(HOP_MARKERS)],
                    color=HOP_COLORS[ci % len(HOP_COLORS)],
                    linewidth=2, markersize=7, label=f'hop={hop}')
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.set_xlabel('Patch Size', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Oversmoothing vs. Patch Size — ' + title_suffix, fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {filename}')


# ─── 3. 热力图：每个数据集单独一张，行=hop，列=patch ───────────────────────
def plot_heatmap_per_dataset(records_by_ds, metric, cmap, title_suffix, filename_prefix):
    for ds in DATASETS:
        ds_rec = records_by_ds.get(ds, [])
        if not ds_rec:
            continue
        hops = sorted(set(r['hop'] for r in ds_rec))
        patches = sorted(set(r['patch'] for r in ds_rec))
        mat = np.full((len(hops), len(patches)), np.nan)
        for r in ds_rec:
            i, j = hops.index(r['hop']), patches.index(r['patch'])
            mat[i, j] = r[metric]
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(mat, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(patches)))
        ax.set_xticklabels(patches, fontsize=10)
        ax.set_yticks(range(len(hops)))
        ax.set_yticklabels(hops, fontsize=10)
        ax.set_xlabel('Patch Size', fontsize=11)
        ax.set_ylabel('Max 2nd-hop (hop)', fontsize=11)
        ax.set_title('{} — {}'.format(DATASET_LABELS.get(ds, ds), title_suffix), fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(hops)):
            for j in range(len(patches)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center', fontsize=9)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f'{filename_prefix}_{ds}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  已保存: {filename_prefix}_{ds}.png')


# ─── 4. 趋势文字总结（基于 cos_sim 与 l2_dist 随 hop/patch 的单调性）──────────
def summarize_trends(records_by_ds):
    lines = []
    lines.append('=' * 60)
    lines.append('四数据集过平滑趋势总结')
    lines.append('指标：cos_sim_mean ↑ 越过平滑；l2_dist_mean ↓ 越过平滑')
    lines.append('=' * 60)

    for ds in DATASETS:
        ds_rec = records_by_ds.get(ds, [])
        if not ds_rec:
            lines.append(f'\n【{ds}】无数据')
            continue
        lines.append(f'\n【{DATASET_LABELS.get(ds, ds)}】')

        # 随 hop 增大
        hop_trends_cos, hop_trends_l2 = [], []
        for patch in sorted(set(r['patch'] for r in ds_rec)):
            pts = sorted([r for r in ds_rec if r['patch'] == patch], key=lambda x: x['hop'])
            if len(pts) < 2:
                continue
            hops = [r['hop'] for r in pts]
            cos_vals = [r['cos_sim_mean'] for r in pts]
            l2_vals = [r['l2_dist_mean'] for r in pts]
            # 整体趋势：最后值减第一个值
            cos_delta = cos_vals[-1] - cos_vals[0]
            l2_delta = l2_vals[-1] - l2_vals[0]
            hop_trends_cos.append((patch, cos_delta))
            hop_trends_l2.append((patch, l2_delta))
        if hop_trends_cos:
            avg_cos = np.mean([d for _, d in hop_trends_cos])
            avg_l2 = np.mean([d for _, d in hop_trends_l2])
            if avg_cos > 0.02:
                lines.append(f'  随二阶邻居数(hop)↑: cos_sim 整体上升 → 过平滑加重')
            elif avg_cos < -0.02:
                lines.append(f'  随二阶邻居数(hop)↑: cos_sim 整体下降 → 过平滑减轻')
            else:
                lines.append(f'  随二阶邻居数(hop)↑: cos_sim 变化不大')
            if avg_l2 < -0.5:
                lines.append(f'  随二阶邻居数(hop)↑: L2距离整体下降 → 过平滑加重')
            elif avg_l2 > 0.5:
                lines.append(f'  随二阶邻居数(hop)↑: L2距离整体上升 → 过平滑减轻')
            else:
                lines.append(f'  随二阶邻居数(hop)↑: L2距离变化不大')

        # 随 patch 增大
        patch_trends_cos, patch_trends_l2 = [], []
        for hop in sorted(set(r['hop'] for r in ds_rec)):
            pts = sorted([r for r in ds_rec if r['hop'] == hop], key=lambda x: x['patch'])
            if len(pts) < 2:
                continue
            cos_vals = [r['cos_sim_mean'] for r in pts]
            l2_vals = [r['l2_dist_mean'] for r in pts]
            patch_trends_cos.append((hop, cos_vals[-1] - cos_vals[0]))
            patch_trends_l2.append((hop, l2_vals[-1] - l2_vals[0]))
        if patch_trends_cos:
            avg_cos = np.mean([d for _, d in patch_trends_cos])
            avg_l2 = np.mean([d for _, d in patch_trends_l2])
            if avg_cos > 0.02:
                lines.append(f'  随 Patch Size↑: cos_sim 整体上升 → 过平滑加重')
            elif avg_cos < -0.02:
                lines.append(f'  随 Patch Size↑: cos_sim 整体下降 → 过平滑减轻')
            else:
                lines.append(f'  随 Patch Size↑: cos_sim 变化不大')
            if avg_l2 < -0.5:
                lines.append(f'  随 Patch Size↑: L2距离整体下降 → 过平滑加重')
            elif avg_l2 > 0.5:
                lines.append(f'  随 Patch Size↑: L2距离整体上升 → 过平滑减轻')
            else:
                lines.append(f'  随 Patch Size↑: L2距离变化不大')

    lines.append('\n' + '=' * 60)
    return '\n'.join(lines)


def main():
    if not os.path.isfile(JSON_PATH):
        print(f'未找到 {JSON_PATH}，请先运行 analyze_oversmoothing_final_layer.py 生成数据。')
        return
    records = load_records()
    records = filter_four_datasets(records)
    if not records:
        print('没有找到四数据集（CanParl, USLegis, uci, wikipedia）的记录。')
        return

    records_by_ds = {}
    for r in records:
        ds = r['dataset']
        records_by_ds.setdefault(ds, []).append(r)

    print('生成「随二阶邻居采样数(hop)增大」趋势图...')
    plot_trend_vs_hop(
        records_by_ds,
        'cos_sim_mean',
        'Mean Pairwise Cosine Similarity (↑ 越过平滑)',
        'Cosine Similarity',
        'trend_vs_hop_cos_sim.png'
    )
    plot_trend_vs_hop(
        records_by_ds,
        'l2_dist_mean',
        'Mean Pairwise L2 Distance (↓ 越过平滑)',
        'L2 Distance',
        'trend_vs_hop_l2_dist.png'
    )

    print('生成「随 Patch Size 增大」趋势图...')
    plot_trend_vs_patch(
        records_by_ds,
        'cos_sim_mean',
        'Mean Pairwise Cosine Similarity (↑ 越过平滑)',
        'Cosine Similarity',
        'trend_vs_patch_cos_sim.png'
    )
    plot_trend_vs_patch(
        records_by_ds,
        'l2_dist_mean',
        'Mean Pairwise L2 Distance (↓ 越过平滑)',
        'L2 Distance',
        'trend_vs_patch_l2_dist.png'
    )

    print('生成每数据集热力图...')
    plot_heatmap_per_dataset(
        records_by_ds, 'cos_sim_mean', 'YlOrRd',
        'Cosine Similarity (↑ 越过平滑)', 'heatmap_cos_sim'
    )
    plot_heatmap_per_dataset(
        records_by_ds, 'l2_dist_mean', 'YlOrRd_r',
        'L2 Distance (↓ 越过平滑)', 'heatmap_l2_dist'
    )

    summary = summarize_trends(records_by_ds)
    print(summary)
    with open(os.path.join(OUT_DIR, 'trend_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'\n趋势总结已写入: {OUT_DIR}/trend_summary.txt')


if __name__ == '__main__':
    main()
