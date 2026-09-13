import argparse
import sys
import torch
import os


def build_run_name_suffix(args, include_batch: bool = True) -> str:
    """拼进 save/load 路径的公共后缀（不含 seed / 任务前缀）。

    与 train_link_prediction 命名规则对齐，供训练/评估/节点分类统一使用，
    避免不同脚本各自拼接导致路径对不上。
    DyGFormerD 会自动带上 dim{rwpe_dim}，从而区分不同 RWPE 预计算维度。
    """
    parts = []
    if getattr(args, 'max_input_sequence_length', None) is not None:
        parts.append(f"seq{args.max_input_sequence_length}")
    if getattr(args, 'patch_size', None) is not None:
        parts.append(f"patch{args.patch_size}")
    if args.model_name in ('DyGFormerTW', 'DyGFormerFF'):
        if getattr(args, 'max_second_hop_length', 0):
            parts.append(f"hop{args.max_second_hop_length}")
    if include_batch:
        if getattr(args, 'batch_size', None) is not None:
            parts.append(f"bs{args.batch_size}")
        if getattr(args, 'gradient_accumulation_steps', 1) > 1:
            parts.append(f"accum{args.gradient_accumulation_steps}")
    if args.model_name == 'DyGFormerD':
        parts.append(f"dim{getattr(args, 'rwpe_dim', 8)}")
    if getattr(args, 'exp_suffix', None):
        parts.append(args.exp_suffix)
    return '_'.join(parts)


def get_link_prediction_args(is_evaluation: bool = False):
    """
    Get the args for the link prediction task.
    """
    parser = argparse.ArgumentParser(description="Interface for the link prediction task")
    
    # Ensure parameters are only defined once
    existing_args = [action.dest for action in parser._actions]
    if 'channel_embedding_dim' not in existing_args:
        parser.add_argument('--channel_embedding_dim', type=int, default=32, help='dimension of each channel embedding')
    if 'num_heads' not in existing_args:
        parser.add_argument('--num_heads', type=int, default=4, help='number of heads used in attention layer')
    if 'num_kv_heads' not in existing_args:
        parser.add_argument('--num_kv_heads', type=int, help='number of key-value heads for GQA (if not specified, use MHA with num_kv_heads=num_heads)')
     # Add task_name parameter
    parser.add_argument('--task_name', type=str, default='link_prediction', choices=['link_prediction', 'node_classification'],
                        help='task to perform, either link_prediction or node_classification')
    # Other parameters...
    parser.add_argument('--model_name', type=str,                         choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerFR', 'DyGFormerRF', 'DyGFormerRFRF', 'DyGFormerRRF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'DyGFormerRS', 'DyGFormerTW', 'DyGFormerCP', 'DyGFormerCR', 'DyGFormerDEG', 'DyGFormerPC', 'DyGFormerRC', 'DIFFDyG','PathDyG', 'NPTDyG', 'NPTSTDyG','DeformableDyG','CARDDyG', 'OCARDDyG', 'CCARDDyG', 'TIMERDyG', 'STDyGFormer', 'STDIFFDyG', 'LDIFFDyG', 'PDyGFormer', 'DyGFormerPF', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPC', 'DyGFormerWSDMPF', 'DyGFormerWSDMR', 'DyGFormerGPR', 'DyGFormerHDSE', 'DyGFormerSPSE', 'DyGFormerRRWP', 'DyGFormerX', 'DyGFormerXR', 'DyGFormerWSDMPX', 'DyGFormerCC'],
                        help='name of the model, note that EdgeBank is only applicable for evaluation')
    # DyGFormerHDSE 专用参数
    parser.add_argument('--max_spd', type=int, default=30,
                        help='DyGFormerHDSE: SPD 截断上限（含），embedding 表大小=max_spd+1')
    parser.add_argument('--max_dist', type=int, default=10,
                        help='DyGFormerHDSE: 社区图距离截断上限（含），embedding 表大小=max_dist+1')
    parser.add_argument('--struct_embed_dim', type=int, default=32,
                        help='DyGFormerHDSE: 结构嵌入维度（CateFeatureEmbedding 内部）')
    parser.add_argument('--hdse_community_method', type=str, default='metis',
                        choices=['metis', 'louvain', 'girvan_newman', 'spectral'],
                        help='DyGFormerHDSE: 已废弃（局部子图内固定用 greedy_modularity_communities），保留仅为兼容旧命令行')
    parser.add_argument('--max_temp_struct_nodes', type=int, default=200,
                        help='DyGFormerHDSE: 单样本诱导子图最大节点数，超过则该样本不加结构偏置')
    parser.add_argument('--spse_dist_len', type=int, default=2,
                        help='DyGFormerSPSE: K（建议 2，最小 2）')
    parser.add_argument('--spse_root_nodes_fact', type=float, default=0.075,
                        help='DyGFormerSPSE: R，root 比例，常用约 0.05～0.1（默认取区间中点 0.075）')
    parser.add_argument('--spse_min_root_nodes', type=int, default=4,
                        help='DyGFormerSPSE: root 数下限；与 max(N\'*R) 联用')
    parser.add_argument('--spse_max_root_nodes', type=int, default=0,
                        help='DyGFormerSPSE: root 数上限；0 表示不限制')
    parser.add_argument('--spse_max_bfs_depth', type=int, default=0,
                        help='DyGFormerSPSE: D_DFS（max_bfs_depth），默认 0')
    parser.add_argument('--spse_max_tries_per_depth', type=int, default=1,
                        help='DyGFormerSPSE: N（max_tries_per_depth），默认 1')
    parser.add_argument('--spse_use_full_spse', action='store_true', default=False,
                        help='DyGFormerSPSE: 使用原 get_simple_paths_count（慢）；默认关闭，改用 A 与 A² 的 walk 计数（快）')
    parser.add_argument('--spse_max_subgraph_nodes', type=int, default=200,
                        help='DyGFormerSPSE: src∪dst 诱导子图节点数上限，超过则跳过该样本 SPSE（bias 为 0）；0 表示不限制')
    parser.add_argument('--spse_precomputed_dir', type=str, default=None,
                        help='DyGFormerSPSE: 预计算 SPSE 数据目录（如 processed_data/wikipedia/spse_precomputed_ps4_dl8）；'
                             '提供后正样本直接查表，不再在线计算，dist_len 自动从 meta.json 读取')
    # DyGFormerD RWPE 专用参数
    parser.add_argument('--rwpe_dim', type=int, default=8,
                        help='DyGFormerD: RWPE 维度（须与预计算一致）')
    parser.add_argument('--rwpe_max_subgraph_size', type=int, default=512,
                        help='DyGFormerD: RWPE 子图 unique 节点数上限（须与预计算一致）')
    parser.add_argument('--rwpe_precomputed_dir', type=str, default=None,
                        help='DyGFormerD: 预计算 RWPE 目录。推荐 node_t：'
                             'processed_data/<ds>/rwpe_node_t_seq512_dim8_ms512（正负均按 (node,t) 查表，miss 懒缓存）；'
                             '旧版 edge_id：rwpe_precomputed_*（仅正样本查表，存在分布不对称）')
    parser.add_argument('--rwpe_lazy_load', action='store_true', default=False,
                        help='DyGFormerD: 懒加载预计算 RWPE chunk（大数据集）')
    # DyGFormerST SPD（最短路径距离）专用参数
    parser.add_argument('--spd_max_subgraph_size', type=int, default=512,
                        help='DyGFormerST: SPD 子图 unique 节点数上限（须与预计算一致），默认与 max_input_sequence_length 一致')
    parser.add_argument('--spd_precomputed_dir', type=str, default=None,
                        help='DyGFormerST: 预计算 SPD 目录（如 processed_data/<ds>/spd_precomputed_modelfn_seq512_ps4_ms512），'
                             'edge_id 键：只覆盖正样本，miss 在线回退；负样本 (edge_ids=None) 始终全部在线计算')
    parser.add_argument('--spd_lazy_load', action='store_true', default=False,
                        help='DyGFormerST: 懒加载预计算 SPD chunk（大数据集）')
    # DyGFormerRRWP 专用参数
    parser.add_argument('--rrwp_walk_length', type=int, default=8,
                        help='DyGFormerRRWP: 随机游走步数 K，pe = [I, T, T^2, ..., T^{K-1}]，特征维为 K')
    parser.add_argument('--rrwp_add_identity', action='store_true', default=True,
                        help='DyGFormerRRWP: 是否在 pe 中包含恒等矩阵 I（k=0 项），默认开启')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of gradient accumulation steps before optimizer update, used to simulate larger batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, e.g. 0')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--max_input_sequence_length', type=int, default=512, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--exp_suffix', type=str, default=None,
                        help='optional extra suffix for save/load paths to avoid overwriting '
                             '(e.g. dim24); DyGFormerD already auto-appends dim{rwpe_dim}')
    parser.add_argument('--max_second_hop_length', type=int, default=None, help='max 2-hop neighbors per node (DyGFormerTW only)')
    parser.add_argument('--max_degree', type=int, default=512,
                        help='DyGFormerDEG: max degree for temporal degree embedding table (degrees above are clamped)')

    try:
        args = parser.parse_args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        parser.print_help()
        sys.exit()
    if args.task_name == 'link_prediction':
       print("Running link prediction task...")
    # 执行链路预测任务的逻辑
    elif args.task_name == 'node_classification':
        print("Running node classification task...")
    # 执行节点分类任务的逻辑
    if args.model_name == 'EdgeBank':
        assert is_evaluation, 'EdgeBank is only applicable for evaluation!'

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args):
    """
    Load the best configurations for the link prediction task.
    """
    if args.model_name == 'TGAT':
        args.num_heads = 2
        args.num_layers = 2
        args.time_feat_dim = 16
        args.dropout = 0.1
        args.learning_rate = 0.0001
        args.weight_decay = 1e-5
        args.patience = 5
    elif args.model_name == 'DyGFormer':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'DyGFormerHDSE':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # HDSE 结构偏置相关
        args.max_spd = getattr(args, 'max_spd', 30)
        args.max_dist = getattr(args, 'max_dist', 10)
        args.struct_embed_dim = getattr(args, 'struct_embed_dim', 32)
        args.hdse_community_method = getattr(args, 'hdse_community_method', 'metis')
    elif args.model_name == 'DyGFormerTW':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if getattr(args, 'max_second_hop_length', None) is None:
            args.max_second_hop_length = 5
    elif args.model_name == 'DyGFormerF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # 保留命令行传入的 num_kv_heads（如果有的话）
        if not hasattr(args, 'num_kv_heads') or args.num_kv_heads is None:
            args.num_kv_heads = None  # 默认使用 MHA
    elif args.model_name == 'DyGFormerFF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if not hasattr(args, 'num_kv_heads') or args.num_kv_heads is None:
            args.num_kv_heads = None
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerFF 维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
        if args.channel_embedding_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerFF (RoPE) 要求 channel_embedding_dim ({args.channel_embedding_dim}) "
                f"能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerRF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32  
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.max_input_sequence_length = 512
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerRF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerRRF':
        args.num_heads = 2
        args.num_layers = 1
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 2
        args.dropout = 0.3
        args.learning_rate = 0.00005
        args.weight_decay = 1e-4
        args.patience = 10
        args.max_input_sequence_length = 256
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerRRF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerRFRF':
        # DyGFormerRFRF: 四通道、一阶邻居、RoPE、差分注意力，无 patch
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.max_input_sequence_length = 128
        args.dropout = 0.1
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerRFRF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerFR':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.max_input_sequence_length = 512
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerFR维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerPF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        
        # 验证维度约束
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerPF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMP':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerWSDMP维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'NPTWSDMP':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"NPTWSDMP维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMPC':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerWSDMPC维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMPF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerWSDMPF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMR':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 2
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerWSDMR维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'PDyGFormer':
        # PDyGFormer 与 DyGFormer 维度一致，使用相同的基础配置
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # PDyGFormer 特定参数
        args.use_multi_scale = getattr(args, 'use_multi_scale', True)  # 默认使用多尺度
        args.patch_size_list = getattr(args, 'patch_size_list', [8, 6, 4, 2])  # 默认多尺度配置
    elif args.model_name in ['DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'DyGFormerRS']:
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        
        # DyGFormerD 特定参数
        if args.model_name == 'DyGFormerD':
            args.use_lpu = getattr(args, 'use_lpu', True)
            args.use_rpe = getattr(args, 'use_rpe', True)
            args.rwpe_dim = getattr(args, 'rwpe_dim', 8)
            args.rwpe_max_subgraph_size = getattr(args, 'rwpe_max_subgraph_size', 512)
            args.rwpe_precomputed_dir = getattr(args, 'rwpe_precomputed_dir', None)
            args.rwpe_lazy_load = getattr(args, 'rwpe_lazy_load', False)

        # DyGFormerST 特定参数
        if args.model_name == 'DyGFormerST':
            args.spd_max_subgraph_size = getattr(args, 'spd_max_subgraph_size', 512)
            args.spd_precomputed_dir = getattr(args, 'spd_precomputed_dir', None)
            args.spd_lazy_load = getattr(args, 'spd_lazy_load', False)
    elif args.model_name == 'STDyGFormer':
        args.num_heads = 4
        args.num_layers = 3
        args.channel_embedding_dim = args.channel_embedding_dim or 32  # 保留命令行参数
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-4
        args.patience = 10
    elif args.model_name == 'DIFFDyG':
        args.num_heads = 4  
        args.num_layers = 2
        args.channel_embedding_dim =  32  
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4  # 增大weight_decay来提升inductive性能
        args.patience = 20
    elif args.model_name == 'STDIFFDyG':
        args.num_heads = 12  # 保留命令行参数
        args.num_layers = 3
        args.channel_embedding_dim =  32  
        args.time_feat_dim = 16
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-3  # 增大weight_decay来提升inductive性能
        args.patience = 10
    elif args.model_name == 'PathDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
       
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'NPTDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'NPTSTDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'DeformableDyG':
        args.num_layers = 2
        args.num_heads = 4
        args.channel_embedding_dim = 32
      
        args.dropout = 0.2
        
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
        else:
            args.max_input_sequence_length = 32
        
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    elif args.model_name == 'CARDDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'OCARDDyG':
        # OCARDDyG: Optimized CARD for dynamic graphs with LayerNorm and dynamic EMA
        args.channel_embedding_dim = 32
        args.d_model = args.channel_embedding_dim
        args.num_heads = 4
        args.num_layers = 2
        args.e_layers = 2
        args.dropout = 0.2
        args.weight_decay = 1e-4
        args.learning_rate = 0.0001
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.patch_len = 4
        args.stride = 1
        args.dp_rank = 8  # dynamic projection rank
        args.alpha = 0.9  # EMA decay factor
        args.merge_size = 1  # head merge size
    elif args.model_name == 'CCARDDyG':
        # CCARDDyG: CARD for dynamic graphs (LayerNorm, dynamic EMA, no positional encoding)
        args.channel_embedding_dim = 32
        args.d_model = args.channel_embedding_dim
        args.num_heads = 4
        args.num_layers = 2
        args.e_layers = 2
        args.dropout = 0.2
        args.weight_decay = 1e-4
        args.learning_rate = 0.0001
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.patch_len = 4
        args.stride = 1
        args.dp_rank = 8
        args.alpha = 0.9
        args.merge_size = 1
        args.d_ff = None
        args.momentum = 0.1
    elif args.model_name == 'TIMERDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name in ['DyGFormerCP', 'DyGFormerCR']:
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if args.model_name == 'DyGFormerCR':
            attention_dim = 4 * args.channel_embedding_dim
            if (attention_dim // args.num_heads) % 2 != 0:
                raise ValueError(
                    f"DyGFormerCR: (4*channel_embedding_dim)/num_heads 必须为偶数，"
                    f"当前 attention_dim={attention_dim}, head_dim={attention_dim//args.num_heads}"
                )
    elif args.model_name == 'DyGFormerDEG':
        # 5 通道；MultiheadAttention 要求 (5 * channel_embedding_dim) % num_heads == 0
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.max_degree = getattr(args, 'max_degree', 512) or 512
        attention_dim = 5 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerDEG: (5*channel_embedding_dim) 必须能被 num_heads 整除，"
                f"当前 attention_dim={attention_dim}, num_heads={args.num_heads}"
            )
    elif args.model_name == 'DyGFormerPC':
        # DyGFormerPC: CARD + Co-occurrence + PositionalEmbedding (类似 DyGFormerCP，但使用 OCARDDyG 的 CARD)
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # CARD 特定参数
        args.dp_rank = 8  # dynamic projection rank
        args.alpha = 0.9  # EMA decay factor
        args.merge_size = 1  # head merge size
        args.d_ff = None  # feed-forward dimension (默认: 4 * channel_embedding_dim)
        args.momentum = 0.1  # momentum (LayerNorm 不使用，但保留兼容性)
    elif args.model_name == 'DyGFormerRC':
        # DyGFormerRC: CARD + Co-occurrence + RoPE (类似 DyGFormerCR，但使用 OCARDDyG 的 CARD)
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # CARD 特定参数
        args.dp_rank = 8  # dynamic projection rank
        args.alpha = 0.9  # EMA decay factor
        args.merge_size = 1  # head merge size
        args.d_ff = None  # feed-forward dimension (默认: 4 * channel_embedding_dim)
        args.momentum = 0.1  # momentum (LayerNorm 不使用，但保留兼容性)
        # RoPE 维度检查：确保 head_dim 是偶数
        head_dim = args.channel_embedding_dim // args.num_heads
        if head_dim % 2 != 0:
            # 调整 channel_embedding_dim 使其满足 RoPE 要求
            original_dim = args.channel_embedding_dim
            # 向上调整到最近的偶数 head_dim
            args.channel_embedding_dim = ((head_dim + 1) // 2) * 2 * args.num_heads
            print(f"警告: DyGFormerRC 需要 head_dim 为偶数。已将 channel_embedding_dim 从 {original_dim} 调整为 {args.channel_embedding_dim}")
    elif args.model_name == 'LDIFFDyG':
        # LDIFFDyG 使用 5 个通道和差分注意力
        # 维度约束：(5 * channel_embedding_dim) % (2 * num_heads) == 0
        args.num_heads = 2 # 4个注意力头
        args.num_layers = 2  # 2层Transformer
        # channel_embedding_dim=32 时，5*32=160，160 % (2*4) = 160 % 8 = 0 ✓
        args.channel_embedding_dim = 36
        args.time_feat_dim = 100  # 时间特征维度
        args.max_input_sequence_length = 128  # 最大序列长度
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # LDIFFDyG 特有参数
        args.use_second_hop = getattr(args, 'use_second_hop', False)  # 默认不使用二阶邻居
        args.num_neighbors = getattr(args, 'num_neighbors', 20)  # 一阶邻居数量
        args.num_kv_heads = getattr(args, 'num_kv_heads', None)  # GQA参数，None表示使用标准MHA
    elif args.model_name == 'DyGFormerGPR':
        # DyGFormerGPR: 用 GPA（K-hop PageRank 注意力）替换标准 MultiheadAttention
        # 无整除维度约束，任意 channel_embedding_dim 均可
        args.num_heads = 4         # GPA 内部暂未使用，预留
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # GPR 特有超参数
        args.K_transformer = getattr(args, 'K_transformer', 10)   # 传播跳数
        args.alpha = getattr(args, 'alpha', 0.1)                   # PageRank 衰减因子
    elif args.model_name == 'DyGFormerSPSE':
        # 标准 MultiheadAttention，约束: 4 * channel_embedding_dim % num_heads == 0
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.spse_dist_len = getattr(args, 'spse_dist_len', 2)
        args.spse_root_nodes_fact = getattr(args, 'spse_root_nodes_fact', 0.075)
        args.spse_min_root_nodes = getattr(args, 'spse_min_root_nodes', 4)
        args.spse_max_root_nodes = getattr(args, 'spse_max_root_nodes', 0)
        args.spse_max_bfs_depth = getattr(args, 'spse_max_bfs_depth', 0)
        args.spse_max_tries_per_depth = getattr(args, 'spse_max_tries_per_depth', 1)
        args.spse_use_full_spse = getattr(args, 'spse_use_full_spse', False)
        args.spse_max_subgraph_nodes = getattr(args, 'spse_max_subgraph_nodes', 200)
        args.spse_precomputed_dir = getattr(args, 'spse_precomputed_dir', None)
    elif args.model_name == 'DyGFormerRRWP':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.rrwp_walk_length = getattr(args, 'rrwp_walk_length', 8)
        args.rrwp_add_identity = getattr(args, 'rrwp_add_identity', True)
    elif args.model_name == 'DyGFormerX':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if args.channel_embedding_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerX: channel_embedding_dim ({args.channel_embedding_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerXR':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        token_head_dim = (4 * args.channel_embedding_dim) // args.num_heads
        if token_head_dim % 4 != 0:
            raise ValueError(
                f"DyGFormerXR: token_head_dim ({token_head_dim}) 必须能被 4 整除，"
                f"当前 4*channel_embedding_dim={4*args.channel_embedding_dim}, num_heads={args.num_heads}"
            )
    elif args.model_name == 'DyGFormerWSDMPX':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if args.channel_embedding_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerWSDMPX: channel_embedding_dim ({args.channel_embedding_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerCC':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.dp_rank = 2
        args.alpha = 0.5
        args.merge_size = 1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")







def get_node_classification_args():
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia', choices=['wikipedia', 'reddit'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerFR', 'DyGFormerRF', 'DyGFormerRFRF', 'DyGFormerRRF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'DyGFormerRS', 'DyGFormerTW', 'DyGFormerCP', 'DyGFormerCR', 'DyGFormerDEG', 'DyGFormerPC', 'DyGFormerRC','DIFFDyG', 'PathDyG','NPTDyG', 'NPTSTDyG','DeformableDyG','CARDDyG', 'OCARDDyG', 'CCARDDyG', 'TIMERDyG', 'STDyGFormer', 'STDIFFDyG', 'LDIFFDyG', 'PDyGFormer', 'DyGFormerPF', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPC', 'DyGFormerWSDMPF', 'DyGFormerWSDMR', 'DyGFormerGPR', 'DyGFormerHDSE', 'DyGFormerSPSE', 'DyGFormerRRWP', 'DyGFormerX', 'DyGFormerXR', 'DyGFormerWSDMPX', 'DyGFormerCC'])
    # DyGFormerHDSE 专用参数（节点分类任务）
    parser.add_argument('--max_spd', type=int, default=30,
                        help='DyGFormerHDSE: SPD 截断上限（含），embedding 表大小=max_spd+1')
    parser.add_argument('--max_dist', type=int, default=10,
                        help='DyGFormerHDSE: 社区图距离截断上限（含），embedding 表大小=max_dist+1')
    parser.add_argument('--struct_embed_dim', type=int, default=32,
                        help='DyGFormerHDSE: 结构嵌入维度（CateFeatureEmbedding 内部）')
    parser.add_argument('--hdse_community_method', type=str, default='metis',
                        choices=['metis', 'louvain', 'girvan_newman', 'spectral'],
                        help='DyGFormerHDSE: 已废弃（局部子图内固定用 greedy_modularity_communities），保留仅为兼容旧命令行')
    parser.add_argument('--max_temp_struct_nodes', type=int, default=200,
                        help='DyGFormerHDSE: 单样本诱导子图最大节点数，超过则该样本不加结构偏置')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, e.g. 0')
    parser.add_argument('--pretrained_model_name', type=str, help='Name of the pretrained model to load (e.g., DIFFDyG_seed1)')
    parser.add_argument('--use_lp_pretrained_backbone', action='store_true', default=False,
                        help='whether to initialize node classification backbone from a pretrained link prediction checkpoint')
    parser.add_argument('--lp_pretrained_model_name', type=str, default=None,
                        help='explicit link prediction checkpoint name to load, e.g. DyGFormer_seed0')
    parser.add_argument('--node_cls_train_mode', type=str, default='classifier_only',
                        choices=['classifier_only', 'finetune', 'random_finetune'],
                        help='classifier_only freezes LP-pretrained backbone; finetune trains LP-pretrained backbone and classifier together; '
                             'random_finetune trains randomly initialized backbone and classifier together')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=32, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=512, help='maximal length of the input sequence of each node')
    # NPTDyG特定参数
    parser.add_argument('--mask_prob', type=float, default=0.15, help='NPTDyG: 遮蔽概率')
    parser.add_argument('--augmentation_mask_prob', type=float, default=0.15, help='NPTDyG: 数据增强时的遮蔽概率')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--start_seed', type=int, default=0, help='starting seed index (inclusive), evaluates seeds [start_seed, start_seed+num_runs)')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--exp_suffix', type=str, default=None,
                        help='optional extra suffix for save/load paths to avoid overwriting '
                             '(e.g. dim24); DyGFormerD already auto-appends dim{rwpe_dim}')
    parser.add_argument('--max_second_hop_length', type=int, default=None, help='max 2-hop neighbors per node (DyGFormerTW only)')
    parser.add_argument('--max_degree', type=int, default=512,
                        help='DyGFormerDEG: max degree for temporal degree embedding table (degrees above are clamped)')
    # DyGFormerSPSE 专用参数（节点分类任务，须与 LP 训练时保持一致）
    parser.add_argument('--spse_dist_len', type=int, default=2,
                        help='DyGFormerSPSE: SPSE 距离向量长度，须与 LP 训练时一致')
    parser.add_argument('--spse_precomputed_dir', type=str, default=None,
                        help='DyGFormerSPSE: 预计算 SPSE 数据目录，须与 LP 训练时一致')
    # DyGFormerD RWPE（须与 LP 训练时一致）
    parser.add_argument('--rwpe_dim', type=int, default=8,
                        help='DyGFormerD: RWPE 维度，须与 LP 训练时一致')
    parser.add_argument('--rwpe_max_subgraph_size', type=int, default=512,
                        help='DyGFormerD: RWPE 子图上限，须与 LP 训练时一致')
    parser.add_argument('--rwpe_precomputed_dir', type=str, default=None,
                        help='DyGFormerD: 预计算 RWPE 目录，须与 LP 训练时一致；'
                             '推荐 rwpe_node_t_*（正负 (node,t) 统一查表）')
    parser.add_argument('--rwpe_lazy_load', action='store_true', default=False,
                        help='DyGFormerD: 懒加载预计算 RWPE chunk')
    # DyGFormerST SPD（须与 LP 训练时一致）
    parser.add_argument('--spd_max_subgraph_size', type=int, default=512,
                        help='DyGFormerST: SPD 子图上限，须与 LP 训练时一致')
    parser.add_argument('--spd_precomputed_dir', type=str, default=None,
                        help='DyGFormerST: 预计算 SPD 目录，须与 LP 训练时一致')
    parser.add_argument('--spd_lazy_load', action='store_true', default=False,
                        help='DyGFormerST: 懒加载预计算 SPD chunk')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of gradient accumulation steps (default 1 = no accumulation)')

    try:
        args = parser.parse_args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in ['wikipedia', 'reddit'], f'Wrong value for dataset_name {args.dataset_name}!'
    if args.load_best_configs:
        load_node_classification_best_configs(args=args)

    return args


def load_node_classification_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node classification task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_heads = 2
        args.num_layers = 2
        args.time_feat_dim = 16
        args.dropout = 0.1
        args.learning_rate = 0.0001
        args.weight_decay = 1e-5
        args.patience = 5
    elif args.model_name == 'DyGFormer':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'DyGFormerHDSE':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.max_spd = getattr(args, 'max_spd', 30)
        args.max_dist = getattr(args, 'max_dist', 10)
        args.struct_embed_dim = getattr(args, 'struct_embed_dim', 32)
        args.hdse_community_method = getattr(args, 'hdse_community_method', 'metis')
    elif args.model_name == 'DyGFormerTW':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if getattr(args, 'max_second_hop_length', None) is None:
            args.max_second_hop_length = 5
    elif args.model_name == 'DyGFormerF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # 保留命令行传入的 num_kv_heads（如果有的话）
        if not hasattr(args, 'num_kv_heads') or args.num_kv_heads is None:
            args.num_kv_heads = None  # 默认使用 MHA
    elif args.model_name == 'DyGFormerFF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if not hasattr(args, 'num_kv_heads') or args.num_kv_heads is None:
            args.num_kv_heads = None
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerFF 维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
        if args.channel_embedding_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerFF (RoPE) 要求 channel_embedding_dim ({args.channel_embedding_dim}) "
                f"能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerRF':
        
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32  
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.max_input_sequence_length = 512
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerRF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerRRF':
        args.num_heads = 2
        args.num_layers = 1
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 2
        args.dropout = 0.3
        args.learning_rate = 0.00005
        args.weight_decay = 1e-4
        args.patience = 10
        args.max_input_sequence_length = 256
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerRRF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerRFRF':
        # DyGFormerRFRF: 四通道、一阶邻居、RoPE、差分注意力，无 patch
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.max_input_sequence_length = 128
        args.dropout = 0.1
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerRFRF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerFR':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.max_input_sequence_length = 512
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerFR维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerPF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # 验证维度约束
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerPF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMP':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerWSDMP维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'NPTWSDMP':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"NPTWSDMP维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMPC':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerWSDMPC维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMPF':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerWSDMPF维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerWSDMR':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 2
        attention_dim = 4 * args.channel_embedding_dim
        if attention_dim % (2 * args.num_heads) != 0:
            raise ValueError(
                f"DyGFormerWSDMR维度配置错误: 4 * channel_embedding_dim ({attention_dim}) "
                f"必须能被 2 * num_heads ({2 * args.num_heads}) 整除"
            )
    elif args.model_name == 'PDyGFormer':
        # PDyGFormer 与 DyGFormer 维度一致，使用相同的基础配置
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # PDyGFormer 特定参数
        args.use_multi_scale = getattr(args, 'use_multi_scale', True)  # 默认使用多尺度
        args.patch_size_list = getattr(args, 'patch_size_list', [8, 6, 4, 2])  # 默认多尺度配置
    elif args.model_name in ['DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'DyGFormerRS']:
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        
        # DyGFormerD 特定参数
        if args.model_name == 'DyGFormerD':
            args.use_lpu = getattr(args, 'use_lpu', True)
            args.use_rpe = getattr(args, 'use_rpe', True)
            args.rwpe_dim = getattr(args, 'rwpe_dim', 8)
            args.rwpe_max_subgraph_size = getattr(args, 'rwpe_max_subgraph_size', 512)
            args.rwpe_precomputed_dir = getattr(args, 'rwpe_precomputed_dir', None)
            args.rwpe_lazy_load = getattr(args, 'rwpe_lazy_load', False)

        # DyGFormerST 特定参数
        if args.model_name == 'DyGFormerST':
            args.spd_max_subgraph_size = getattr(args, 'spd_max_subgraph_size', 512)
            args.spd_precomputed_dir = getattr(args, 'spd_precomputed_dir', None)
            args.spd_lazy_load = getattr(args, 'spd_lazy_load', False)
    elif args.model_name == 'DyGFormerC':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'STDyGFormer':
        args.num_heads = 4
        args.num_layers = 3
        args.channel_embedding_dim = args.channel_embedding_dim or 32  # 保留命令行参数
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-4
        args.patience = 10
    elif args.model_name == 'DIFFDyG':
        args.num_heads = 4  
        args.num_layers = 2
        args.channel_embedding_dim =  32  
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4  # 增大weight_decay来提升inductive性能
        args.patience = 20
    elif args.model_name == 'STDIFFDyG':
        args.num_heads = 12  # 保留命令行参数
        args.num_layers = 3
        args.channel_embedding_dim =  32  
        args.time_feat_dim = 16
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-3  # 增大weight_decay来提升inductive性能
        args.patience = 10
    elif args.model_name == 'PathDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
       
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'NPTDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'NPTSTDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'DeformableDyG':
        args.num_layers = 2
        args.num_heads = 4
        args.channel_embedding_dim = 32
      
        args.dropout = 0.2
        
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
        else:
            args.max_input_sequence_length = 32
        
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    elif args.model_name == 'CARDDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name == 'OCARDDyG':
        # OCARDDyG: Optimized CARD for dynamic graphs with LayerNorm and dynamic EMA
        args.channel_embedding_dim = 32
        args.d_model = args.channel_embedding_dim
        args.num_heads = 4
        args.num_layers = 2
        args.e_layers = 2
        args.dropout = 0.2
        args.weight_decay = 1e-4
        args.learning_rate = 0.0001
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.patch_len = 4
        args.stride = 1
        args.dp_rank = 8  # dynamic projection rank
        args.alpha = 0.9  # EMA decay factor
        args.merge_size = 1  # head merge size
    elif args.model_name == 'CCARDDyG':
        # CCARDDyG: CARD for dynamic graphs (LayerNorm, dynamic EMA, no positional encoding)
        args.channel_embedding_dim = 32
        args.d_model = args.channel_embedding_dim
        args.num_heads = 4
        args.num_layers = 2
        args.e_layers = 2
        args.dropout = 0.2
        args.weight_decay = 1e-4
        args.learning_rate = 0.0001
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.patch_len = 4
        args.stride = 1
        args.dp_rank = 8
        args.alpha = 0.9
        args.merge_size = 1
        args.d_ff = None
        args.momentum = 0.1
    elif args.model_name == 'TIMERDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
    elif args.model_name in ['DyGFormerCP', 'DyGFormerCR']:
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if args.model_name == 'DyGFormerCR':
            attention_dim = 4 * args.channel_embedding_dim
            if (attention_dim // args.num_heads) % 2 != 0:
                raise ValueError(
                    f"DyGFormerCR: (4*channel_embedding_dim)/num_heads 必须为偶数，"
                    f"当前 attention_dim={attention_dim}, head_dim={attention_dim//args.num_heads}"
                )
    elif args.model_name == 'DyGFormerDEG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.max_degree = getattr(args, 'max_degree', 512) or 512
        attention_dim = 5 * args.channel_embedding_dim
        if attention_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerDEG: (5*channel_embedding_dim) 必须能被 num_heads 整除，"
                f"当前 attention_dim={attention_dim}, num_heads={args.num_heads}"
            )
    elif args.model_name == 'DyGFormerPC':
        # DyGFormerPC: CARD + Co-occurrence + PositionalEmbedding (类似 DyGFormerCP，但使用 OCARDDyG 的 CARD)
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # CARD 特定参数
        args.dp_rank = 8  # dynamic projection rank
        args.alpha = 0.9  # EMA decay factor
        args.merge_size = 1  # head merge size
        args.d_ff = None  # feed-forward dimension (默认: 4 * channel_embedding_dim)
        args.momentum = 0.1  # momentum (LayerNorm 不使用，但保留兼容性)
    elif args.model_name == 'DyGFormerRC':
        # DyGFormerRC: CARD + Co-occurrence + RoPE (类似 DyGFormerCR，但使用 OCARDDyG 的 CARD)
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # CARD 特定参数
        args.dp_rank = 8  # dynamic projection rank
        args.alpha = 0.9  # EMA decay factor
        args.merge_size = 1  # head merge size
        args.d_ff = None  # feed-forward dimension (默认: 4 * channel_embedding_dim)
        args.momentum = 0.1  # momentum (LayerNorm 不使用，但保留兼容性)
        # RoPE 维度检查：确保 head_dim 是偶数
        head_dim = args.channel_embedding_dim // args.num_heads
        if head_dim % 2 != 0:
            # 调整 channel_embedding_dim 使其满足 RoPE 要求
            original_dim = args.channel_embedding_dim
            # 向上调整到最近的偶数 head_dim
            args.channel_embedding_dim = ((head_dim + 1) // 2) * 2 * args.num_heads
            print(f"警告: DyGFormerRC 需要 head_dim 为偶数。已将 channel_embedding_dim 从 {original_dim} 调整为 {args.channel_embedding_dim}")
    elif args.model_name == 'LDIFFDyG':
        # LDIFFDyG 使用 5 个通道和差分注意力
        # 维度约束：(5 * channel_embedding_dim) % (2 * num_heads) == 0
        args.num_heads = 2 # 4个注意力头
        args.num_layers = 2  # 2层Transformer
        # channel_embedding_dim=32 时，5*32=160，160 % (2*4) = 160 % 8 = 0 ✓
        args.channel_embedding_dim = 32
        args.time_feat_dim = 100  # 时间特征维度
        args.max_input_sequence_length = 128  # 最大序列长度
        args.dropout = 0.1
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # LDIFFDyG 特有参数
        args.use_second_hop = getattr(args, 'use_second_hop', False)  # 默认不使用二阶邻居
        args.num_neighbors = getattr(args, 'num_neighbors', 20)  # 一阶邻居数量
        args.num_kv_heads = getattr(args, 'num_kv_heads', None)  # GQA参数，None表示使用标准MHA
    elif args.model_name == 'DyGFormerGPR':
        # DyGFormerGPR: 用 GPA（K-hop PageRank 注意力）替换标准 MultiheadAttention
        # 无整除维度约束，任意 channel_embedding_dim 均可
        args.num_heads = 4         # GPA 内部暂未使用，预留
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        # GPR 特有超参数
        args.K_transformer = getattr(args, 'K_transformer', 10)   # 传播跳数
        args.alpha = getattr(args, 'alpha', 0.1)                   # PageRank 衰减因子
    elif args.model_name == 'DyGFormerSPSE':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim =  32 
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.spse_dist_len = getattr(args, 'spse_dist_len', 2)
        args.spse_root_nodes_fact = getattr(args, 'spse_root_nodes_fact', 0.075)
        args.spse_min_root_nodes = getattr(args, 'spse_min_root_nodes', 4)
        args.spse_max_root_nodes = getattr(args, 'spse_max_root_nodes', 0)
        args.spse_max_bfs_depth = getattr(args, 'spse_max_bfs_depth', 0)
        args.spse_max_tries_per_depth = getattr(args, 'spse_max_tries_per_depth', 1)
        args.spse_use_full_spse = getattr(args, 'spse_use_full_spse', False)
        args.spse_max_subgraph_nodes = getattr(args, 'spse_max_subgraph_nodes', 200)
        args.spse_precomputed_dir = getattr(args, 'spse_precomputed_dir', None)
    elif args.model_name == 'DyGFormerRRWP':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.rrwp_walk_length = getattr(args, 'rrwp_walk_length', 8)
        args.rrwp_add_identity = getattr(args, 'rrwp_add_identity', True)
    elif args.model_name == 'DyGFormerX':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if args.channel_embedding_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerX: channel_embedding_dim ({args.channel_embedding_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerXR':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        token_head_dim = (4 * args.channel_embedding_dim) // args.num_heads
        if token_head_dim % 4 != 0:
            raise ValueError(
                f"DyGFormerXR: token_head_dim ({token_head_dim}) 必须能被 4 整除，"
                f"当前 4*channel_embedding_dim={4*args.channel_embedding_dim}, num_heads={args.num_heads}"
            )
    elif args.model_name == 'DyGFormerWSDMPX':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        if args.channel_embedding_dim % args.num_heads != 0:
            raise ValueError(
                f"DyGFormerWSDMPX: channel_embedding_dim ({args.channel_embedding_dim}) "
                f"必须能被 num_heads ({args.num_heads}) 整除"
            )
    elif args.model_name == 'DyGFormerCC':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 32
        args.patch_size = 4
        args.max_input_sequence_length = 512
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-4
        args.patience = 20
        args.dp_rank = 2
        args.alpha = 0.5
        args.merge_size = 1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
