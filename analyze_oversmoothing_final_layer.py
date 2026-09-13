# -*- coding: utf-8 -*-
"""
DyGFormerTW 最终层节点嵌入过平滑分析脚本
=========================================
自动扫描 saved_models/DyGFormerTW 下所有存在的检查点，
对每个（dataset, hop, patch, seed）配置，
在测试集上提取 output_layer 之后的最终节点嵌入，
计算：
  - 成对余弦相似度均值（↑ → 越过平滑）
  - 成对 L2 距离均值   （↓ → 越过平滑）

结果保存至 ./oversmoothing_final_layer/
"""

import os
import re
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.DataLoader import get_link_prediction_data, get_idx_data_loader
from utils.utils import get_neighbor_sampler, set_random_seed
from models.DyGFormerTW import DyGFormerTW
from models.modules import MergeLayer

# ═══════════════════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 只分析 seed=0（多 seed 会成倍增加运行时间；结论在单 seed 下已有代表性）
TARGET_SEED = 0

# 只关注有多组 hop/patch 对比配置的数据集；单模型数据集跳过
MULTI_CONFIG_DATASETS = {'wikipedia', 'uci', 'CanParl', 'USLegis'}

# 模型超参（与训练时保持一致）
TIME_FEAT_DIM  = 100
CHANNEL_DIM    = 32
NUM_LAYERS     = 2
NUM_HEADS      = 4
DROPOUT        = 0.0
MAX_SEQ_LEN    = 512

# 无 hop/patch 后缀的基础模型默认超参（来自 DyGFormerTW 构造函数默认值）
DEFAULT_HOP    = 32
DEFAULT_PATCH  = 4

# 测试集抽样：每批 200 个交互，最多取 10 批
BATCH_SIZE  = 200
NUM_BATCHES = 10

SAVE_DIR = './oversmoothing_final_layer'
os.makedirs(SAVE_DIR, exist_ok=True)

CKPT_ROOT = './saved_models/DyGFormerTW'

# ═══════════════════════════════════════════════════════════════════════════════
# 自动扫描所有可用检查点
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_model_dir(model_dir_name: str):
    """
    从目录名解析 seed / hop / patch。
    支持格式：
      DyGFormerTW_seed{s}
      DyGFormerTW_seed{s}_hop{h}_patch{p}
      DyGFormerTW_seed{s}_hop{h}         （无 patch 后缀，跳过）
    返回 (seed, hop, patch) 或 None（无法解析时）
    """
    # 带 hop 和 patch
    m = re.fullmatch(r'DyGFormerTW_seed(\d+)_hop(\d+)_patch(\d+)', model_dir_name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # 仅 seed（基础模型）
    m = re.fullmatch(r'DyGFormerTW_seed(\d+)', model_dir_name)
    if m:
        return int(m.group(1)), DEFAULT_HOP, DEFAULT_PATCH
    return None


def discover_checkpoints(seed_filter: int = None):
    """
    扫描 CKPT_ROOT 下的所有 .pkl，返回列表：
      [(dataset, seed, hop, patch, ckpt_path), ...]
    """
    entries = []
    for dataset in sorted(os.listdir(CKPT_ROOT)):
        ds_dir = os.path.join(CKPT_ROOT, dataset)
        if not os.path.isdir(ds_dir):
            continue
        for model_dir in sorted(os.listdir(ds_dir)):
            parsed = _parse_model_dir(model_dir)
            if parsed is None:
                continue
            seed, hop, patch = parsed
            if seed_filter is not None and seed != seed_filter:
                continue
            ckpt_path = os.path.join(ds_dir, model_dir, f'{model_dir}.pkl')
            if os.path.isfile(ckpt_path):
                entries.append((dataset, seed, hop, patch, ckpt_path))
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# 相似度度量
# ═══════════════════════════════════════════════════════════════════════════════

def mean_pairwise_cosine_sim(emb: torch.Tensor) -> float:
    """batch 内所有非对角节点对的平均余弦相似度（↑ → 越过平滑）。"""
    normed = F.normalize(emb.float(), p=2, dim=-1)
    sim_mat = normed @ normed.T
    N = emb.shape[0]
    if N < 2:
        return float('nan')
    mask = ~torch.eye(N, dtype=torch.bool, device=emb.device)
    return sim_mat[mask].mean().item()


def mean_pairwise_l2_dist(emb: torch.Tensor) -> float:
    """batch 内所有非对角节点对的平均 L2 距离（↓ → 越过平滑）。"""
    N = emb.shape[0]
    if N < 2:
        return float('nan')
    diff = emb.float().unsqueeze(0) - emb.float().unsqueeze(1)  # (N, N, D)
    dist_mat = diff.norm(p=2, dim=-1)
    mask = ~torch.eye(N, dtype=torch.bool, device=emb.device)
    return dist_mat[mask].mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# 模型构建与加载
# ═══════════════════════════════════════════════════════════════════════════════

def build_backbone(node_raw_features, edge_raw_features,
                   neighbor_sampler, patch: int, hop: int) -> DyGFormerTW:
    return DyGFormerTW(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=TIME_FEAT_DIM,
        channel_embedding_dim=CHANNEL_DIM,
        patch_size=patch,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_input_sequence_length=MAX_SEQ_LEN,
        device=DEVICE,
        max_second_hop_length=hop,
    )


def load_full_model(node_raw_features, edge_raw_features,
                    neighbor_sampler, patch: int, hop: int,
                    ckpt_path: str) -> DyGFormerTW:
    backbone = build_backbone(node_raw_features, edge_raw_features,
                               neighbor_sampler, patch, hop)
    link_predictor = MergeLayer(
        input_dim1=backbone.node_feat_dim,
        input_dim2=backbone.node_feat_dim,
        hidden_dim=128,
        output_dim=1,
    )
    model = nn.Sequential(backbone, link_predictor)
    state = torch.load(ckpt_path, map_location=DEVICE)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 核心：提取最终嵌入并计算指标
# ═══════════════════════════════════════════════════════════════════════════════

def compute_final_embedding_metrics(backbone: DyGFormerTW,
                                    test_data, test_loader) -> dict:
    """
    对测试集前 NUM_BATCHES 个 batch 提取 output_layer 之后的最终节点嵌入，
    拼接所有 src 和 dst 嵌入，计算成对余弦相似度和 L2 距离。
    """
    all_emb = []
    cos_list, l2_list = [], []

    backbone.eval()
    with torch.no_grad():
        for batch_idx, indices in enumerate(test_loader):
            if batch_idx >= NUM_BATCHES:
                break
            indices = (indices.numpy()
                       if hasattr(indices, 'numpy') else np.array(indices))
            src   = test_data.src_node_ids[indices]
            dst   = test_data.dst_node_ids[indices]
            times = test_data.node_interact_times[indices]

            src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src, dst, times)
            # 每个 batch 内单独计算，避免内存爆炸
            batch_emb = torch.cat([src_emb.detach(), dst_emb.detach()], dim=0)
            cos_list.append(mean_pairwise_cosine_sim(batch_emb))
            l2_list.append(mean_pairwise_l2_dist(batch_emb))
            all_emb.append(batch_emb.cpu())

    cos_arr = np.array([x for x in cos_list if not np.isnan(x)])
    l2_arr  = np.array([x for x in l2_list  if not np.isnan(x)])

    # 跨 batch 全局嵌入（用于补充统计）
    all_emb_cat = torch.cat(all_emb, dim=0)
    global_cos = mean_pairwise_cosine_sim(all_emb_cat.to(DEVICE))
    global_l2  = mean_pairwise_l2_dist(all_emb_cat.to(DEVICE))

    return {
        'cos_sim_mean':   float(np.mean(cos_arr)) if len(cos_arr) else float('nan'),
        'cos_sim_std':    float(np.std(cos_arr))  if len(cos_arr) else float('nan'),
        'l2_dist_mean':   float(np.mean(l2_arr))  if len(l2_arr)  else float('nan'),
        'l2_dist_std':    float(np.std(l2_arr))   if len(l2_arr)  else float('nan'),
        'global_cos_sim': global_cos,
        'global_l2_dist': global_l2,
        'num_nodes':      all_emb_cat.shape[0],
        'num_batches':    len(cos_list),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_COLORS = {
    'wikipedia': '#1f77b4',
    'uci':       '#ff7f0e',
    'CanParl':   '#2ca02c',
    'USLegis':   '#d62728',
}

def _label(hop, patch, is_default=False):
    suffix = ' (base)' if is_default else ''
    return f'hop{hop}/p{patch}{suffix}'


def plot_bar_per_dataset(ds_results: dict, dataset: str):
    """
    对单个 dataset 绘制两个并排柱状图：
      左：cos_sim_mean（误差棒=std；每个 batch 内部方差）
      右：l2_dist_mean
    x 轴 = 不同 (hop, patch) 配置；颜色深浅随 hop 递增。
    """
    items = sorted(ds_results.items(), key=lambda x: (x[0][0], x[0][1]))
    labels   = [_label(h, p) for (h, p), _ in items]
    cos_vals = [r['cos_sim_mean'] for _, r in items]
    cos_stds = [r['cos_sim_std']  for _, r in items]
    l2_vals  = [r['l2_dist_mean'] for _, r in items]
    l2_stds  = [r['l2_dist_std']  for _, r in items]

    n = len(labels)
    x = np.arange(n)
    base_color = DATASET_COLORS.get(dataset, '#888888')

    # 根据 hop 大小生成颜色深浅
    hops = sorted(set(h for (h, _) in ds_results))
    alpha_map = {h: 0.4 + 0.6 * i / max(1, len(hops) - 1)
                 for i, h in enumerate(hops)}
    colors = [plt.matplotlib.colors.to_rgba(base_color,
              alpha=alpha_map[h]) for (h, p), _ in items]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 1.2), 5))
    fig.suptitle(f'{dataset} — Final-Layer Embedding Similarity  '
                 f'(seed={TARGET_SEED}, {NUM_BATCHES} batches × {BATCH_SIZE})',
                 fontsize=12)

    for ax, vals, stds, ylabel, title, arrow in [
        (axes[0], cos_vals, cos_stds,
         'Mean Pairwise Cosine Similarity',
         'Cosine Similarity  (↑ = more over-smooth)', '↑'),
        (axes[1], l2_vals,  l2_stds,
         'Mean Pairwise L2 Distance',
         'L2 Distance  (↓ = more over-smooth)', '↓'),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.errorbar(x, vals, yerr=stds, fmt='none', color='black',
                    capsize=4, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        # 标注数值
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f'bar_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [图表] {path}')


def plot_heatmaps(all_results: list):
    """
    对每个 dataset，以 hop（行）× patch（列）为轴绘制热力图，
    分别展示 cos_sim 和 l2_dist。
    """
    datasets = sorted(set(r['dataset'] for r in all_results))

    for metric, cmap, title_suffix in [
        ('cos_sim_mean', 'YlOrRd',   'Mean Pairwise Cosine Similarity  (↑ = more over-smooth)'),
        ('l2_dist_mean', 'YlOrRd_r', 'Mean Pairwise L2 Distance  (↓ = more over-smooth)'),
    ]:
        ncols = len(datasets)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
        if ncols == 1:
            axes = [axes]
        fig.suptitle(f'Heatmap: {title_suffix}', fontsize=12)

        for ax, ds in zip(axes, datasets):
            ds_res = [r for r in all_results if r['dataset'] == ds]
            hops    = sorted(set(r['hop']   for r in ds_res))
            patches = sorted(set(r['patch'] for r in ds_res))
            mat = np.full((len(hops), len(patches)), np.nan)
            for r in ds_res:
                i = hops.index(r['hop'])
                j = patches.index(r['patch'])
                mat[i, j] = r[metric]

            im = ax.imshow(mat, cmap=cmap, aspect='auto',
                           vmin=np.nanmin(mat), vmax=np.nanmax(mat))
            ax.set_xticks(range(len(patches)))
            ax.set_xticklabels([f'p{p}' for p in patches], fontsize=9)
            ax.set_yticks(range(len(hops)))
            ax.set_yticklabels([f'h{h}' for h in hops], fontsize=9)
            ax.set_xlabel('Patch Size')
            ax.set_ylabel('Max 2nd-hop Length')
            ax.set_title(ds, fontsize=11)
            plt.colorbar(im, ax=ax, shrink=0.85)

            for i in range(len(hops)):
                for j in range(len(patches)):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f'{mat[i, j]:.4f}',
                                ha='center', va='center', fontsize=8,
                                color='black')

        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'heatmap_{metric}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [图表] {path}')


def plot_line_vs_hop(all_results: list):
    """
    对每个 dataset，以 hop 为 x 轴，不同 patch size 为折线，
    展示 cos_sim / l2_dist 随 hop 增大的变化趋势。
    """
    datasets = sorted(set(r['dataset'] for r in all_results))
    PATCH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b']
    PATCH_MARKERS = ['o', 's', '^', 'D', 'v', 'P']

    for metric, ylabel, title_suffix in [
        ('cos_sim_mean', 'Mean Pairwise Cosine Similarity', 'Cosine Similarity vs. Max-2nd-hop'),
        ('l2_dist_mean', 'Mean Pairwise L2 Distance',       'L2 Distance vs. Max-2nd-hop'),
    ]:
        ncols = len(datasets)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=False)
        if ncols == 1:
            axes = [axes]
        fig.suptitle(title_suffix, fontsize=12)

        for ax, ds in zip(axes, datasets):
            ds_res = [r for r in all_results if r['dataset'] == ds]
            patches = sorted(set(r['patch'] for r in ds_res))
            for ci, patch in enumerate(patches):
                pts = sorted([r for r in ds_res if r['patch'] == patch],
                             key=lambda r: r['hop'])
                hops = [r['hop'] for r in pts]
                vals = [r[metric] for r in pts]
                ax.plot(hops, vals,
                        marker=PATCH_MARKERS[ci % len(PATCH_MARKERS)],
                        color=PATCH_COLORS[ci % len(PATCH_COLORS)],
                        linewidth=1.8, markersize=6,
                        label=f'patch={patch}')
            ax.set_title(ds, fontsize=11)
            ax.set_xlabel('Max 2nd-hop Length (hop)')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'line_hop_{metric}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [图表] {path}')


def plot_line_vs_patch(all_results: list):
    """
    对每个 dataset，以 patch 为 x 轴，不同 hop 大小为折线，
    展示 cos_sim / l2_dist 随 patch size 增大的变化趋势。
    """
    datasets = sorted(set(r['dataset'] for r in all_results))
    HOP_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                   '#9467bd', '#8c564b', '#e377c2']
    HOP_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']

    for metric, ylabel, title_suffix in [
        ('cos_sim_mean', 'Mean Pairwise Cosine Similarity', 'Cosine Similarity vs. Patch Size'),
        ('l2_dist_mean', 'Mean Pairwise L2 Distance',       'L2 Distance vs. Patch Size'),
    ]:
        ncols = len(datasets)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=False)
        if ncols == 1:
            axes = [axes]
        fig.suptitle(title_suffix, fontsize=12)

        for ax, ds in zip(axes, datasets):
            ds_res = [r for r in all_results if r['dataset'] == ds]
            hops = sorted(set(r['hop'] for r in ds_res))
            for ci, hop in enumerate(hops):
                pts = sorted([r for r in ds_res if r['hop'] == hop],
                             key=lambda r: r['patch'])
                if len(pts) < 2:
                    continue
                ps   = [r['patch'] for r in pts]
                vals = [r[metric]  for r in pts]
                ax.plot(ps, vals,
                        marker=HOP_MARKERS[ci % len(HOP_MARKERS)],
                        color=HOP_COLORS[ci % len(HOP_COLORS)],
                        linewidth=1.8, markersize=6,
                        label=f'hop={hop}')
            ax.set_title(ds, fontsize=11)
            ax.set_xlabel('Patch Size')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'line_patch_{metric}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [图表] {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f'使用设备: {DEVICE}')

    # 扫描全部 seed=0 的检查点
    all_ckpts = discover_checkpoints(seed_filter=TARGET_SEED)
    # 只保留有多组对比配置的数据集
    all_ckpts = [(ds, s, h, p, path)
                 for (ds, s, h, p, path) in all_ckpts
                 if ds in MULTI_CONFIG_DATASETS]

    print(f'发现 {len(all_ckpts)} 个待分析检查点（seed={TARGET_SEED}，'
          f'数据集：{sorted(set(ds for ds, *_ in all_ckpts))}）\n')

    all_results = []
    summary = {}

    # 按 dataset 分组处理，减少数据重复加载
    datasets = sorted(set(ds for ds, *_ in all_ckpts))
    for dataset in datasets:
        ds_ckpts = [(s, h, p, path)
                    for (ds, s, h, p, path) in all_ckpts
                    if ds == dataset]
        print(f'══ 数据集: {dataset}  ({len(ds_ckpts)} 个配置) ══')

        set_random_seed(TARGET_SEED)

        # 一次性加载数据集特征（所有配置共用）
        (node_raw_features, edge_raw_features,
         full_data, train_data, val_data, test_data,
         new_node_val_data, new_node_test_data) = get_link_prediction_data(
            dataset_name=dataset, val_ratio=0.15, test_ratio=0.15)

        neighbor_sampler = get_neighbor_sampler(
            data=full_data,
            sample_neighbor_strategy='recent',
            seed=TARGET_SEED,
        )

        test_loader = get_idx_data_loader(
            indices_list=list(range(len(test_data.src_node_ids))),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        ds_layer_results = {}   # {(hop, patch): metrics_dict}

        for seed, hop, patch, ckpt_path in sorted(ds_ckpts, key=lambda x: (x[1], x[2])):
            tag = f'hop={hop}  patch={patch}'
            print(f'  [{tag}]  加载: {os.path.basename(ckpt_path)}')

            try:
                model = load_full_model(
                    node_raw_features, edge_raw_features,
                    neighbor_sampler, patch, hop, ckpt_path)
            except Exception as e:
                print(f'    !! 加载失败: {e}')
                continue

            backbone: DyGFormerTW = model[0]
            backbone.set_neighbor_sampler(neighbor_sampler)

            try:
                metrics = compute_final_embedding_metrics(
                    backbone, test_data, test_loader)
            except Exception as e:
                print(f'    !! 推理失败: {e}')
                del model, backbone
                torch.cuda.empty_cache()
                continue

            print(f'    cos_sim={metrics["cos_sim_mean"]:.4f} ± {metrics["cos_sim_std"]:.4f}'
                  f'  |  l2_dist={metrics["l2_dist_mean"]:.4f} ± {metrics["l2_dist_std"]:.4f}'
                  f'  |  nodes={metrics["num_nodes"]}')

            record = {'dataset': dataset, 'seed': seed, 'hop': hop, 'patch': patch}
            record.update(metrics)
            all_results.append(record)
            ds_layer_results[(hop, patch)] = metrics

            key = f'{dataset}_seed{seed}_hop{hop}_patch{patch}'
            summary[key] = metrics

            del model, backbone
            torch.cuda.empty_cache()

        # per-dataset 柱状图
        if ds_layer_results:
            plot_bar_per_dataset(ds_layer_results, dataset)
        print()

    # 跨数据集图表
    if all_results:
        print('正在生成跨数据集图表...')
        plot_heatmaps(all_results)
        plot_line_vs_hop(all_results)
        plot_line_vs_patch(all_results)

    # 保存 JSON 汇总
    json_path = os.path.join(SAVE_DIR, 'oversmoothing_final_layer.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[完成] 所有结果保存至 {SAVE_DIR}/')
    print(f'  oversmoothing_final_layer.json  ← 数值汇总')
    print(f'  bar_{{dataset}}.png              ← 每数据集柱状图')
    print(f'  heatmap_cos_sim_mean.png        ← hop×patch 热力图（余弦相似度）')
    print(f'  heatmap_l2_dist_mean.png        ← hop×patch 热力图（L2 距离）')
    print(f'  line_hop_*.png                  ← 随 hop 增大的趋势折线图')
    print(f'  line_patch_*.png                ← 随 patch 增大的趋势折线图')


if __name__ == '__main__':
    main()
