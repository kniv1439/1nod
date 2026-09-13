"""
使用训练好的DyGFormerF模型在wikipedia数据集上生成节点嵌入的t-SNE可视化

使用方法:
    python visualize_node_embeddings_tsne.py [可选参数]
    
可选参数:
    --seed: 模型seed，默认为0，需要与训练时使用的seed一致
    --dataset_name: 数据集名称，默认为wikipedia
    --model_name: 模型名称，默认为DyGFormerF
    
示例:
    # 使用默认参数（seed=0）
    python visualize_node_embeddings_tsne.py
    
    # 使用seed=1的模型
    python visualize_node_embeddings_tsne.py --seed 1
    
输出:
    - 可视化图像: ./visualizations/DyGFormerF/wikipedia/tsne_visualization_*.png
    - 嵌入数据: ./visualizations/DyGFormerF/wikipedia/embeddings_tsne_*.npz
"""
import logging
import time
import sys
import os
import numpy as np
import warnings
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from models.DyGFormerF import DyGFormerF
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler
from utils.DataLoader import get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args
from models.modules import MLPClassifier

# 导入自定义分类器（与训练脚本保持一致）
class CustomMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super(CustomMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


def get_model_submodule(model, idx):
    """从DataParallel模型中获取子模块"""
    if isinstance(model, nn.DataParallel):
        return model.module[idx]
    else:
        return model[idx]


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # 设置参数
    args = get_node_classification_args()
    args.model_name = 'DyGFormerF'
    args.dataset_name = 'wikipedia'
    args.seed = 0  # 可以根据需要修改，确保与训练时使用的seed一致
    args.load_model_name = f'node_classification_{args.model_name}_seed{args.seed}'
    
    # 可以通过命令行参数覆盖默认值
    # 例如: python visualize_node_embeddings_tsne.py --seed 1 --channel_embedding_dim 64
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/visualize_tsne/", exist_ok=True)
    fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/visualize_tsne/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'开始生成t-SNE可视化，配置: {args}')

    # 加载数据
    logger.info('加载数据集...')
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    logger.info(f'数据集信息: 总交互数={full_data.num_interactions}, 唯一节点数={full_data.num_unique_nodes}')

    # 初始化neighbor sampler
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # 直接从保存的模型文件中读取权重并推断配置
    load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
    model_path = os.path.join(load_model_folder, f"{args.load_model_name}.pkl")
    
    config_inferred = False
    if os.path.exists(model_path):
        logger.info(f'从保存的模型文件读取配置: {model_path}')
        try:
            # 直接加载模型权重文件
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 从多个权重推断配置
            # 方法1: 从 projection_layer.node.weight 推断 channel_embedding_dim
            # 方法2: 从 lambda_q1 推断 head_dim，然后反推 channel_embedding_dim
            inferred_channel_dim = None
            inferred_head_dim = None
            
            # 尝试不同的键名（可能是Sequential模型或DataParallel模型）
            prefix = None
            for test_prefix in ['0.', 'module.0.']:
                if test_prefix + 'projection_layer.node.weight' in checkpoint:
                    prefix = test_prefix
                    break
            
            if prefix is None:
                # 如果没找到，列出所有键帮助调试
                logger.warning('未找到 projection_layer.node.weight，可用的键:')
                for key in list(checkpoint.keys())[:10]:  # 只显示前10个键
                    logger.warning(f'  - {key}')
            else:
                # 从 projection_layer.node.weight 推断 channel_embedding_dim
                proj_key = prefix + 'projection_layer.node.weight'
                if proj_key in checkpoint:
                    weight_shape = checkpoint[proj_key].shape
                    inferred_channel_dim = weight_shape[0]
                    logger.info(f'从键 "{proj_key}" 推断 channel_embedding_dim = {inferred_channel_dim}')
                
                # 从 lambda_q1 推断 head_dim
                lambda_key = prefix + 'transformers.0.multi_head_attention.lambda_q1'
                if lambda_key in checkpoint:
                    lambda_shape = checkpoint[lambda_key].shape
                    inferred_head_dim = lambda_shape[0]
                    logger.info(f'从键 "{lambda_key}" 推断 head_dim = {inferred_head_dim}')
                    
                    # 从 subln.weight 推断 attention_dim
                    # subln.weight 的形状是 [attention_dim // 2]
                    subln_key = prefix + 'transformers.0.multi_head_attention.subln.weight'
                    if subln_key in checkpoint:
                        subln_shape = checkpoint[subln_key].shape
                        inferred_attention_dim = subln_shape[0] * 2
                        logger.info(f'从键 "{subln_key}" 推断 attention_dim = {inferred_attention_dim}')
                        
                        # 从 attention_dim 和 head_dim 反推 num_heads
                        # head_dim = attention_dim // num_heads // 2
                        # 所以: num_heads = attention_dim // (head_dim * 2)
                        inferred_num_heads = inferred_attention_dim // (inferred_head_dim * 2)
                        logger.info(f'从 attention_dim={inferred_attention_dim} 和 head_dim={inferred_head_dim} 反推 num_heads = {inferred_num_heads}')
                        
                        # 更新 num_heads
                        if inferred_num_heads > 0:
                            args.num_heads = inferred_num_heads
                    
                    # 从 head_dim 反推 channel_embedding_dim
                    # head_dim = attention_dim // num_heads // 2
                    # attention_dim = num_channels * channel_embedding_dim
                    # 所以: head_dim = (num_channels * channel_embedding_dim) // num_heads // 2
                    # 因此: channel_embedding_dim = (head_dim * num_heads * 2) // num_channels
                    num_channels = 4
                    inferred_channel_dim_from_head = (inferred_head_dim * args.num_heads * 2) // num_channels
                    logger.info(f'从 head_dim={inferred_head_dim} 和 num_heads={args.num_heads} 反推 channel_embedding_dim = {inferred_channel_dim_from_head}')
                    
                    # 使用从 head_dim 反推的值（更准确）
                    if inferred_channel_dim_from_head > 0:
                        inferred_channel_dim = inferred_channel_dim_from_head
            
            if inferred_channel_dim is not None:
                logger.info(f'使用推断的 channel_embedding_dim = {inferred_channel_dim}')
                args.channel_embedding_dim = inferred_channel_dim
                config_inferred = True
            else:
                logger.error('无法推断 channel_embedding_dim')
                sys.exit(1)
        except Exception as e:
            logger.error(f'无法从模型文件读取配置: {e}')
            logger.error('请手动指定 --channel_embedding_dim 参数')
            sys.exit(1)
    else:
        logger.error(f'模型文件不存在: {model_path}')
        sys.exit(1)

    # 创建模型
    logger.info('创建模型...')
    set_random_seed(seed=args.seed)
    
    # 确保维度满足差分注意力的要求
    # 如果配置是从模型推断的，通常已经满足要求，不需要调整
    num_channels = 4
    attention_dim = num_channels * args.channel_embedding_dim
    if attention_dim % (2 * args.num_heads) != 0:
        if config_inferred:
            logger.warning(f'推断的配置可能不满足差分注意力要求: attention_dim={attention_dim}, 2*num_heads={2*args.num_heads}')
            logger.warning('这可能导致模型创建失败，请检查训练时的配置')
        else:
            original_dim = args.channel_embedding_dim
            required_divisor = 2 * args.num_heads
            while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                args.channel_embedding_dim += 1
            logger.info(f"调整channel_embedding_dim从{original_dim}到{args.channel_embedding_dim}以满足差分注意力要求")
    
    dynamic_backbone = DyGFormerF(
        node_raw_features=node_raw_features, 
        edge_raw_features=edge_raw_features, 
        neighbor_sampler=full_neighbor_sampler,
        time_feat_dim=args.time_feat_dim, 
        channel_embedding_dim=args.channel_embedding_dim, 
        patch_size=args.patch_size,
        num_layers=args.num_layers, 
        num_heads=args.num_heads, 
        dropout=args.dropout,
        max_input_sequence_length=args.max_input_sequence_length, 
        device=args.device
    )
    
    output_dim = node_raw_features.shape[1]
    node_classifier = CustomMLPClassifier(input_dim=output_dim, hidden_dim=128, dropout=args.dropout)
    
    model = nn.Sequential(dynamic_backbone, node_classifier)
    logger.info(f'模型创建完成，参数量: {get_parameter_sizes(model) * 4 / 1024 / 1024:.4f} MB')

    # 加载训练好的模型
    if not os.path.exists(load_model_folder):
        logger.error(f'模型文件夹不存在: {load_model_folder}')
        logger.error('请确保模型已经训练并保存。')
        sys.exit(1)
    
    checkpoint_path = os.path.join(load_model_folder, f"{args.load_model_name}.pkl")
    if not os.path.exists(checkpoint_path):
        logger.error(f'模型文件不存在: {checkpoint_path}')
        sys.exit(1)
    
    logger.info(f'正在加载模型权重: {checkpoint_path}')
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试加载权重，使用strict=False允许部分匹配
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info('✅ 模型权重加载成功（完全匹配）')
        except RuntimeError as e:
            # 如果完全匹配失败，尝试部分匹配
            logger.warning(f'⚠️  模型配置不完全匹配，尝试部分加载...')
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.warning(f'  未匹配的键（将使用随机初始化）: {len(missing_keys)} 个')
                    if len(missing_keys) <= 10:
                        logger.warning(f"    示例: {missing_keys[:5]}")
                if unexpected_keys:
                    logger.warning(f'  意外的键（已忽略）: {len(unexpected_keys)} 个')
                logger.warning('⚠️  模型权重部分加载成功（部分匹配）')
            except RuntimeError as e2:
                # 即使strict=False也失败，说明尺寸完全不匹配
                logger.error(f'❌ 无法加载模型权重（尺寸完全不匹配）: {e2}')
                logger.error('⚠️  将使用随机初始化的模型（结果可能不准确）')
                logger.error('   建议：检查模型配置参数是否与训练时一致')
    except Exception as e:
        logger.error(f'加载模型权重时出错: {e}')
        import traceback
        traceback.print_exc()
        logger.error('将使用随机初始化的模型（结果可能不准确）')

    model = model.to(args.device)
    model = nn.DataParallel(model)
    model.eval()
    
    # 设置neighbor sampler
    get_model_submodule(model, 0).set_neighbor_sampler(full_neighbor_sampler)

    logger.info('开始提取节点嵌入...')
    
    # 获取所有唯一节点
    all_nodes = sorted(list(full_data.unique_node_ids))
    logger.info(f'总共有 {len(all_nodes)} 个唯一节点')
    
    # 为每个节点找到最后一次交互的时间和标签
    # 使用字典存储每个节点的所有交互信息
    node_interactions = defaultdict(list)
    node_last_interaction = {}
    
    for i in range(len(full_data.src_node_ids)):
        src_node = full_data.src_node_ids[i]
        dst_node = full_data.dst_node_ids[i]
        interact_time = full_data.node_interact_times[i]
        label = full_data.labels[i]
        
        # 记录所有交互
        node_interactions[src_node].append({'time': interact_time, 'label': label})
        node_interactions[dst_node].append({'time': interact_time, 'label': label})
        
        # 更新源节点的最后一次交互
        if src_node not in node_last_interaction or interact_time > node_last_interaction[src_node]['time']:
            node_last_interaction[src_node] = {'time': interact_time, 'label': label}
        
        # 更新目标节点的最后一次交互
        if dst_node not in node_last_interaction or interact_time > node_last_interaction[dst_node]['time']:
            node_last_interaction[dst_node] = {'time': interact_time, 'label': label}
    
    # 获取节点标签
    # 方法1: 使用最后一次交互的标签（默认）
    # 方法2: 使用最常见的标签（可以通过参数选择）
    use_most_common_label = False  # 设置为True使用最常见的标签
    
    node_labels = []
    node_times = []
    for node_id in all_nodes:
        if node_id in node_last_interaction:
            if use_most_common_label and node_id in node_interactions:
                # 使用最常见的标签
                labels = [interaction['label'] for interaction in node_interactions[node_id]]
                most_common_label = max(set(labels), key=labels.count)
                node_labels.append(most_common_label)
            else:
                # 使用最后一次交互的标签
                node_labels.append(node_last_interaction[node_id]['label'])
            node_times.append(node_last_interaction[node_id]['time'])
        else:
            # 如果节点没有交互记录，使用默认值
            node_labels.append(0)
            node_times.append(0.0)
    
    node_labels = np.array(node_labels)
    node_times = np.array(node_times)
    
    logger.info(f'节点标签分布: 类别0={np.sum(node_labels == 0)}, 类别1={np.sum(node_labels == 1)}')
    
    # 批量提取节点嵌入
    batch_size = 200
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(all_nodes), batch_size):
            batch_nodes = all_nodes[i:i+batch_size]
            batch_times = node_times[i:i+batch_size]
            
            # 为了计算嵌入，我们需要源节点和目标节点
            # 对于节点分类任务，我们可以使用节点自身作为源和目标
            batch_src_node_ids = np.array(batch_nodes)
            batch_dst_node_ids = np.array(batch_nodes)  # 使用相同的节点作为目标
            batch_node_interact_times = batch_times
            
            # 获取节点嵌入
            batch_src_node_embeddings, batch_dst_node_embeddings = \
                get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                )
            
            # 使用源节点嵌入（也可以使用平均或其他方式）
            batch_embeddings = batch_src_node_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f'已处理 {i + len(batch_nodes)}/{len(all_nodes)} 个节点')
    
    # 合并所有嵌入
    all_embeddings = np.vstack(all_embeddings)
    logger.info(f'节点嵌入形状: {all_embeddings.shape}')
    
    # 使用t-SNE降维
    logger.info('开始t-SNE降维...')
    # 根据节点数量调整perplexity
    n_samples = len(all_embeddings)
    perplexity = min(30, max(5, n_samples // 100))  # perplexity应该在5到30之间，且小于样本数
    logger.info(f'使用perplexity={perplexity} (样本数={n_samples})')
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    logger.info('t-SNE降维完成')
    
    # 可视化
    logger.info('生成可视化图像...')
    
    # 获取唯一标签和类别数
    unique_labels_plot = np.unique(node_labels)
    n_classes = len(unique_labels_plot)
    
    # 统计每个类别的节点数量
    label_counts_plot = [np.sum(node_labels == label) for label in unique_labels_plot]
    
    # 设置颜色映射：两个类别分别用蓝色和橙色
    if n_classes == 2:
        colors = ['#1f77b4', '#ff7f0e']  # 类别0：蓝色，类别1：橙色
    else:
        # 如果类别数不是2，使用默认颜色方案
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # 创建图形
    plt.figure(figsize=(12, 8))  # 12英寸宽，8英寸高
    
    # 点样式设置
    if n_classes == 2:
        # 找到少数类和多数类
        minority_idx = 0 if label_counts_plot[0] < label_counts_plot[1] else 1
        majority_idx = 1 - minority_idx
        
        # 设置点大小 - 两种类别使用相同大小
        point_size = 30  # 统一的点大小
        sizes = [point_size, point_size]
        
        # 先绘制多数类（在底层），再绘制少数类（在上层）
        for idx in [majority_idx, minority_idx]:
            label = unique_labels_plot[idx]
            mask = node_labels == label
            
            # 使用合适的alpha值，确保都能看到
            alpha_val = 0.6 if idx == majority_idx else 1.0
            
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=colors[idx],                                    # 颜色
                        label=f'Class {int(label)} ({label_counts_plot[idx]} nodes)',  # 图例标签
                        alpha=alpha_val,                                  # 透明度（多数类0.6，少数类1.0）
                        s=sizes[idx],                                     # 点大小（30）
                        edgecolors='black',                               # 黑色边框
                        linewidths=0.5,                                   # 边框宽度
                        zorder=idx+1)                                     # 绘制顺序（多数类先画，少数类后画）
    else:
        sizes = [20] * n_classes  # 每个类别点大小为20
        for i, label in enumerate(unique_labels_plot):
            mask = node_labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=colors[i],                                      # 颜色
                        label=f'Class {int(label)} ({label_counts_plot[i]} nodes)',  # 图例标签
                        alpha=0.7,                                        # 透明度0.7
                        s=sizes[i],                                       # 点大小20
                        edgecolors='black',                               # 黑色边框
                        linewidths=0.8)                                   # 边框宽度0.8
    
    # 标题和坐标轴
    plt.title(f'{args.dataset_name} Dataset - Node Classification t-SNE Visualization',
              fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)  # 图例位置自动选择最佳位置
    plt.grid(True, alpha=0.3)            # 显示网格，透明度0.3
    
    # 保存图像
    save_dir = f"./visualizations/{args.model_name}/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"tsne_visualization_{args.load_model_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300: 高分辨率, bbox_inches='tight': 自动裁剪空白边缘
    logger.info(f'可视化图像已保存到: {save_path}')
    
    # 也保存嵌入数据以便后续使用
    save_data_path = os.path.join(save_dir, f"embeddings_tsne_{args.load_model_name}.npz")
    np.savez(save_data_path, 
             embeddings_2d=embeddings_2d,
             embeddings_original=all_embeddings,
             node_labels=node_labels,
             node_ids=all_nodes)
    logger.info(f'嵌入数据已保存到: {save_data_path}')
    
    logger.info('完成！')
