import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn
import ast

from models.DIFFDyG import DIFFDyG
from models.PathDyG import PathDyG
from models.NPTDyG import NPTDyG
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.DeformableDyG import DeformableDyG
from models.CARDDyG import CARDDyG
from models.modules import MergeLayer
from models.TIMERDyG import TIMERDyG
from models.STDyGFormer import STDyGFormer
from models.STDIFFDyG import STDIFFDyG
from models.RLDyGFormer import RLDyGFormer
from models.RLDIFFDyG import RLDIFFDyG
from models.LDyGFormer import LDyGFormer
from models.LDIFFDyG import LDIFFDyG

from models.EMADIFFDyG import EMADIFFDyG
from models.SRLDIFFDyG import SRLDIFFDyG
from models.HRLDIFFDyG import HRLDIFFDyG

from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.graph import NeighborFinder  # 新增

# ----------------------------------------------------------------------
# helper: 初始化态注意力分析函数
# ----------------------------------------------------------------------

def run_initialization_analysis(model_name: str, model: nn.Module, seq_length: int = 64, batch_size: int = 1):
    """
    运行初始化态注意力分析，用于分析稳定秩和谱间隙
    
    Args:
        model_name: 模型名称
        model: 模型实例
        seq_length: 序列长度
        batch_size: batch大小
        
    Returns:
        dict: 分析结果字典，如果失败则返回None
    """
    try:
        # 获取模型的backbone部分
        backbone = get_model_submodule(model, 0)
        
        # 检查模型是否支持初始化态分析
        if hasattr(backbone, 'run_initialization_analysis'):
            logger = logging.getLogger()
            logger.info(f'开始对 {model_name} 进行初始化态分析...')
            
            # 运行分析
            results = backbone.run_initialization_analysis(
                seq_length=seq_length,
                batch_size=batch_size
            )
            
            logger.info(f'{model_name} 初始化态分析完成')
            return results
        else:
            logging.getLogger().warning(f'{model_name} 不支持初始化态分析')
            return None
            
    except Exception as e:
        logging.getLogger().error(f'初始化态分析失败: {str(e)}')
        return None

# ----------------------------------------------------------------------
# helper: build NeighborFinder from Data (复用 train_link_prediction 中的实现)
# ----------------------------------------------------------------------

def build_neighbor_finder(data, uniform=False, seed=0):
    """Convert Data object to NeighborFinder (k-hop sampler)."""
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    adj_list = [[] for _ in range(max_node_id + 1)]
    for s, d, e, t in zip(data.src_node_ids,
                          data.dst_node_ids,
                          data.edge_ids,
                          data.node_interact_times):
        adj_list[s].append((d, e, t))
        adj_list[d].append((s, e, t))
    return NeighborFinder(adj_list, balance=1.0, seed=seed, uniform=uniform)

from evaluate_models_utils import evaluate_model_link_prediction, get_model_submodule
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args  # 修改：使用链路预测参数

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # 修改：使用链路预测参数而非节点分类参数
    args = get_link_prediction_args(is_evaluation=True)
    
    print(f"Running link prediction task...")
    
    # 获取链路预测数据
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # 初始化邻居采样器 (一阶)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # 为 ST 系列模型构造 full-graph k-hop NeighborFinder
    full_neighbor_finder = build_neighbor_finder(full_data, uniform=args.sample_neighbor_strategy == 'uniform', seed=1)

    # 获取数据加载器
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # 设置负采样器
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    test_metric_all_runs = []
    new_node_test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_result_name = f'evaluate_link_prediction_{args.model_name}_seed{args.seed}'

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # 创建模型
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # 计算时间偏移，用于JODIE模型
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device,
                                         enable_stable_rank_collection=True,
                                         stable_rank_save_dir=f"results/stable_rank_evaluation/{args.dataset_name}/DyGFormer_seed{run}")
        elif args.model_name == 'LDyGFormer':
            dynamic_backbone = LDyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)  
        elif args.model_name == 'RLDyGFormer':
            dynamic_backbone = RLDyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'DIFFDyG':
            # 特殊处理：确保channel_embedding_dim能被num_heads整除
            if hasattr(args, 'channel_embedding_dim') and hasattr(args, 'num_heads'):
                if args.channel_embedding_dim % args.num_heads != 0:
                    old_dim = args.channel_embedding_dim
                    # 调整channel_embedding_dim为num_heads的倍数
                    args.channel_embedding_dim = (args.channel_embedding_dim // args.num_heads) * args.num_heads
                    if args.channel_embedding_dim == 0:
                        args.channel_embedding_dim = args.num_heads * 64  # 默认使用较大的值
                    logger.warning(f"自动调整channel_embedding_dim从{old_dim}到{args.channel_embedding_dim}，以确保它能被num_heads({args.num_heads})整除")
            
            dynamic_backbone = DIFFDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                enable_stable_rank_collection=True,
                stable_rank_save_dir=f"results/stable_rank_evaluation/{args.dataset_name}/DIFFDyG_seed{run}"
            )
        elif args.model_name == 'LDIFFDyG':
            dynamic_backbone = LDIFFDyG(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers,
                                        num_heads=args.num_heads, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'RLDIFFDyG':
            dynamic_backbone = RLDIFFDyG(node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device)
        elif args.model_name == 'HRLDIFFDyG':
            dynamic_backbone = RLDIFFDyG(node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device)
        elif args.model_name == 'SRLDIFFDyG':
            dynamic_backbone = RLDIFFDyG(node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device)
        elif args.model_name == 'EMADIFFDyG':
           dynamic_backbone = EMADIFFDyG(node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device)
        elif args.model_name == 'RLSDIFFDyG':
            dynamic_backbone = RLDIFFDyG(node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device)
        elif args.model_name == 'NPTDyG':
            dynamic_backbone = NPTDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )
        elif args.model_name == 'PathDyG':
            dynamic_backbone = PathDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )
        elif args.model_name == 'DeformableDyG':
            dynamic_backbone = DeformableDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )
        elif args.model_name == 'CARDDyG':
            dynamic_backbone = CARDDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                config=args,  # 传递整个args作为config
                device=args.device
            )
        elif args.model_name == 'TIMERDyG':
            dynamic_backbone = TIMERDyG(
                config=args,
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_channels=4,
                task="link_prediction",
                device=args.device
            )
        elif args.model_name == 'STDyGFormer':
            # 解析num_neighbors参数
            num_neighbors = getattr(args, 'num_neighbors', [10, 5])
            if isinstance(num_neighbors, str):
                num_neighbors = ast.literal_eval(num_neighbors)
                
            dynamic_backbone = STDyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_finder=full_neighbor_finder,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                k_hop=getattr(args, 'k_hop', 2),
                num_neighbors=num_neighbors
            )
        elif args.model_name == 'STDIFFDyG':
            # 解析num_neighbors参数
            num_neighbors = getattr(args, 'num_neighbors', [10, 5])
            if isinstance(num_neighbors, str):
                num_neighbors = ast.literal_eval(num_neighbors)
                
            dynamic_backbone = STDIFFDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_finder=full_neighbor_finder,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                k_hop=getattr(args, 'k_hop', 2),
                num_neighbors=num_neighbors
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        # 创建link_predictor - 根据不同模型使用不同的输入维度
        if args.model_name in ['DyGFormer', 'TIMERDyG', 'STDyGFormer', 'RLDyGFormer','LDyGFormer']:
            input_dim1 = dynamic_backbone.channel_embedding_dim * dynamic_backbone.patch_size
            input_dim2 = dynamic_backbone.channel_embedding_dim * dynamic_backbone.patch_size
            link_predictor = MergeLayer(
                input_dim1=input_dim1,
                input_dim2=input_dim2,
                hidden_dim=128,
                output_dim=1
             )
        elif args.model_name in ['DeformableDyG', 'PathDyG', 'CARDDyG']:
            # PathDyG需要特殊处理
            if args.model_name == 'PathDyG':
                # 对PathDyG使用特定的维度
                try:
                    # 尝试动态获取维度
                    actual_dim = getattr(dynamic_backbone, 'output_dim', None)
                    if actual_dim is None:
                        actual_dim = dynamic_backbone.channel_embedding_dim * 4
                except:
                    actual_dim = dynamic_backbone.channel_embedding_dim * 4
                
                link_predictor = MergeLayer(
                    input_dim1=actual_dim,
                    input_dim2=actual_dim,
                    hidden_dim=128,
                    output_dim=1
                )
            else:  # 其他模型
                out_dim = dynamic_backbone.channel_embedding_dim * 4
                link_predictor = MergeLayer(
                    input_dim1=out_dim,
                    input_dim2=out_dim,
                    hidden_dim=128,
                    output_dim=1
                )
        elif args.model_name == 'NPTDyG':
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
             )
        elif args.model_name in ['DIFFDyG', 'STDIFFDyG', 'RLDIFFDyG','LDIFFDyG','RLSDIFFDyG','EMADIFFDyG','SRLDIFFDyG','HRLDIFFDyG']:
            # DIFFDyG和STDIFFDyG使用node_feat_dim作为输出维度
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
            )
        else:
            # 其他模型默认使用node_feat_dim
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
             )

        # 组合模型
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # 加载已训练的模型
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}/"
        
        # 尝试加载原始训练配置
        config_path = os.path.join(load_model_folder, "train_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    train_config = json.load(f)
                    logger.info(f"从{config_path}加载训练配置")
                    
                    # 应用关键参数
                    for key in ['channel_embedding_dim', 'num_heads', 'num_layers', 'time_feat_dim', 'max_input_sequence_length']:
                        if key in train_config and hasattr(args, key):
                            old_val = getattr(args, key)
                            setattr(args, key, train_config[key])
                            logger.info(f"参数 {key} 从 {old_val} 更新为 {train_config[key]}")
            except Exception as e:
                logger.warning(f"加载训练配置失败: {str(e)}")
        else:
            # 如果配置文件不存在，尝试记录参数
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                config_to_save = {k: v for k, v in vars(args).items() if not k.startswith('_')}
                with open(config_path, 'w') as f:
                    json.dump(config_to_save, f, indent=4)
                logger.info(f"当前参数已保存到{config_path}")
            except Exception as e:
                logger.warning(f"保存当前配置失败: {str(e)}")
        
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                      save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        
        logger.info(f'尝试从 {load_model_folder} 加载模型...')
        try:
            early_stopping.load_checkpoint(model)
            logger.info('模型加载成功!')
        except Exception as e:
            logger.error(f'模型加载失败: {str(e)}')
            if run == 0:  # 如果第一次运行就失败，则终止程序
                sys.exit(1)
            else:
                logger.warning("跳过此次运行")
                continue

        model = model.to(args.device)
        model = nn.DataParallel(model)
        print('Using', torch.cuda.device_count(), 'GPUs')

        loss_func = nn.BCELoss()

        # 评估最佳模型
        logger.info(f'在 {args.dataset_name} 数据集上评估模型性能...')

        # 评估测试集性能
        eval_sampler = full_neighbor_finder if args.model_name in ['STDyGFormer', 'STDIFFDyG'] else full_neighbor_sampler

        
# --- spectral metrics CSV path settings ---
_runs_dir = os.path.join("runs", "metrics", args.model_name, args.dataset_name)
os.makedirs(_runs_dir, exist_ok=True)
save_csv_path = os.path.join(_runs_dir, "test_samples.csv")
phase_label = "test"
test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                  model=model,
                                                                  neighbor_sampler=eval_sampler,
                                                                  evaluate_idx_data_loader=test_idx_data_loader,
                                                                  evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                  evaluate_data=test_data,
                                                                  loss_func=loss_func,
                                                                  num_neighbors=args.num_neighbors,
                                                                  time_gap=args.time_gap,
                                                                  collect_attention=(args.model_name in ['DIFFDyG', 'DyGFormer']))

        # 评估新节点测试集性能
        
# --- spectral metrics CSV for new-node test ---
save_csv_path = os.path.join(_runs_dir, "new_node_test_samples.csv")
phase_label = "new_node_test"
new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                    model=model,
                                                                                    neighbor_sampler=eval_sampler,
                                                                                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                    evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                    evaluate_data=new_node_test_data,
                                                                                    loss_func=loss_func,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap,
                                                                                    collect_attention=False)  # 新节点测试不需要收集注意力

        # 运行初始化态注意力分析
        logger.info('运行初始化态注意力分析...')
        try:
            init_analysis_results = run_initialization_analysis(
                model_name=args.model_name,
                model=model,
                seq_length=min(64, args.max_input_sequence_length),  # 使用较小的序列长度进行分析
                batch_size=2
            )
            
            if init_analysis_results:
                logger.info('初始化态分析完成，统计结果:')
                for key, value in init_analysis_results['statistics']['overall'].items():
                    logger.info(f'  {key}: {value}')
                
                # 保存分析结果
                analysis_save_path = os.path.join(save_result_folder, f"{args.save_result_name}_init_analysis.json")
                with open(analysis_save_path, 'w') as f:
                    json.dump(init_analysis_results, f, indent=2, default=str)
                logger.info(f'初始化态分析结果已保存到 {analysis_save_path}')
                
        except Exception as e:
            logger.warning(f'初始化态分析失败: {str(e)}')

        # 存储当前运行的评估指标
        test_metric_dict = {}
        new_node_test_metric_dict = {}

        logger.info(f'测试集损失: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'测试集 {metric_name}: {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'新节点测试集损失: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'新节点测试集 {metric_name}: {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # 避免日志重叠
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # 保存模型结果
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }

        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        
        logger.info(f'结果已保存到 {save_result_path}')
        
        # 显示稳定秩分析结果
        try:
            backbone = get_model_submodule(model, 0)
            if hasattr(backbone, 'stable_rank_collector') and backbone.stable_rank_collector is not None:
                analysis = backbone.get_stable_rank_analysis(min_sequence_length=5)
                if analysis and 'error' not in analysis:
                    print(f"\n{'='*100}")
                    print(f"STABLE RANK ANALYSIS - {args.model_name} on {args.dataset_name}")
                    print(f"{'='*100}")
                    
                    print(f"总样本数: {analysis['total_samples']}")
                    
                    if 'rank_collapse' in analysis:
                        collapse_info = analysis['rank_collapse']
                        print(f"\n秩坍缩分析:")
                        print(f"  坍缩率: {collapse_info['collapse_rate']:.4f}")
                        print(f"  坍缩样本数: {collapse_info['collapsed_samples']}")
                        print(f"  坍缩阈值: {collapse_info['threshold']}")
                    
                    if 'sequence_analysis' in analysis and analysis['sequence_analysis']:
                        print(f"\n按序列长度分析:")
                        for seq_info in analysis['sequence_analysis']:
                            print(f"  T={seq_info['sequence_length']}: 平均稳定秩={seq_info['mean_stable_rank']:.4f} ± {seq_info['std_stable_rank']:.4f}")
                    
                    if 'layer_analysis' in analysis and analysis['layer_analysis']:
                        print(f"\n按层数分析:")
                        for layer_info in analysis['layer_analysis']:
                            print(f"  第{layer_info['layer_idx']+1}层: 平均稳定秩={layer_info['mean_stable_rank']:.4f} ± {layer_info['std_stable_rank']:.4f}")
                    
                    # 理论验证结果
                    print(f"\n理论验证:")
                    if args.model_name == 'DIFFDyG':
                        print("  - 差分注意力机制验证")
                        if 'rank_collapse' in analysis:
                            rate = analysis['rank_collapse']['collapse_rate']
                            if rate < 0.3:
                                print(f"  ✓ 优秀: 秩坍缩率 {rate:.3f} 显著低于标准注意力预期")
                            elif rate < 0.5:
                                print(f"  ✓ 良好: 秩坍缩率 {rate:.3f} 低于标准注意力预期")
                            else:
                                print(f"  ⚠ 待改进: 秩坍缩率 {rate:.3f} 未达到理论预期")
                    elif args.model_name == 'DyGFormer':
                        print("  - 标准注意力机制验证")
                        if 'rank_collapse' in analysis:
                            rate = analysis['rank_collapse']['collapse_rate']
                            print(f"  - 秩坍缩率: {rate:.3f} (标准注意力的典型表现)")
                    
                    print(f"{'='*100}")
                    
                    # 保存分析结果
                    backbone.save_stable_rank_results(suffix=f"_evaluation_{args.dataset_name}")
                    logger.info("稳定秩分析结果已保存")
                else:
                    print(f"\n{args.model_name} 稳定秩分析: 暂无足够数据")
        except Exception as e:
            logger.warning(f"稳定秩分析失败: {e}")

    # 如果有多次运行，输出所有运行的平均指标
    if args.num_runs > 1:
        logger.info(f'{args.num_runs}次运行的平均指标:')
        
        for metric_name in test_metric_all_runs[0].keys():
            logger.info(f'测试集 {metric_name}: {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'平均测试集 {metric_name}: {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        
        for metric_name in new_node_test_metric_all_runs[0].keys():
            logger.info(f'新节点测试集 {metric_name}: {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
            logger.info(f'平均新节点测试集 {metric_name}: {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                        f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
