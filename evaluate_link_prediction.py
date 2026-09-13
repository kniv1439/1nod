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
from models.NPTSTDyG import NPTSTDyG
from models.NPTWSDMP import NPTWSDMP
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.DyGFormerTW import DyGFormerTW
# DyGFormerD is imported lazily when needed to avoid dgl dependency issues
from models.DyGFormerP import DyGFormerP
from models.DyGFormerR import DyGFormerR
from models.DyGFormerC import DyGFormerC
from models.DyGFormerCP import DyGFormerCP
from models.DyGFormerDEG import DyGFormerDEG
from models.DyGFormerCR import DyGFormerCR
from models.DyGFormerPC import DyGFormerPC
from models.DyGFormerRC import DyGFormerRC
from models.DyGFormerRF import DyGFormerRF
from models.DyGFormerRFRF import DyGFormerRFRF
from models.DyGFormerFR import DyGFormerFR
from models.DyGFormerRRF import DyGFormerRRF
from models.DyGFormerPF import DyGFormerPF
from models.DyGFormerWSDMP import DyGFormerWSDMP
from models.DyGFormerWSDMPC import DyGFormerWSDMPC
from models.DyGFormerWSDMPF import DyGFormerWSDMPF
from models.DyGFormerWSDMR import DyGFormerWSDMR
from models.DeformableDyG import DeformableDyG
from models.CARDDyG import CARDDyG
from models.oldCARDDyG import OCARDDyG
from models.CCARDDyG import CCARDDyG
from models.PDyGFormer import PDyGFormer
from models.modules import MergeLayer
from models.TIMERDyG import TIMERDyG

from models.RLDyGFormer import RLDyGFormer
from models.RLDIFFDyG import RLDIFFDyG
from models.LDyGFormer import LDyGFormer
from models.LDIFFDyG import LDIFFDyG
from models.DyGFormerGPR import DyGFormerGPR
from models.DyGFormerHDSE import DyGFormerHDSE
from models.DyGFormerSPSE import DyGFormerSPSE
from models.DyGFormerRRWP import DyGFormerRRWP
from models.DyGFormerX import DyGFormerX
from models.DyGFormerXR import DyGFormerXR
from models.DyGFormerWSDMPX import DyGFormerWSDMPX





from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, get_directed_neighbor_sampler, NegativeEdgeSampler
from utils.graph import NeighborFinder
 # 新增

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
from utils.load_configs import get_link_prediction_args, build_run_name_suffix  # 修改：使用链路预测参数

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

    # DyGFormerDEG requires directed neighbor sampler to distinguish in-degree and out-degree
    if args.model_name == 'DyGFormerDEG':
        full_neighbor_sampler = get_directed_neighbor_sampler(data=full_data,
                                                              sample_neighbor_strategy=args.sample_neighbor_strategy,
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
        suffix = build_run_name_suffix(args, include_batch=True)
        if suffix:
            args.load_model_name += '_' + suffix
            args.save_result_name += '_' + suffix

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
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'DyGFormerTW':
            dynamic_backbone = DyGFormerTW(
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
                device=args.device,
                max_second_hop_length=getattr(args, 'max_second_hop_length', 32),
            )
        elif args.model_name == 'DyGFormerF':
            from models.DyGFormerF import DyGFormerF
            dynamic_backbone = DyGFormerF(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features, 
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads,
                num_kv_heads=getattr(args, 'num_kv_heads', None),
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=args.device
            )
        elif args.model_name == 'DyGFormerFF':
            from models.DyGFormerFF import DyGFormerFF
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"自动调整 channel_embedding_dim 从 {old_dim} 到 {args.channel_embedding_dim}，"
                    f"以满足 attention_dim % (2*num_heads) == 0"
                )
            if args.channel_embedding_dim % args.num_heads != 0:
                old_dim = args.channel_embedding_dim
                args.channel_embedding_dim = (args.channel_embedding_dim // args.num_heads) * args.num_heads
                if args.channel_embedding_dim == 0:
                    args.channel_embedding_dim = args.num_heads
                logger.warning(
                    f"自动调整 channel_embedding_dim 从 {old_dim} 到 {args.channel_embedding_dim}，"
                    f"以满足 RoPE 要求 (channel_embedding_dim % num_heads == 0)"
                )
            dynamic_backbone = DyGFormerFF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_kv_heads=getattr(args, 'num_kv_heads', None),
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )
        elif args.model_name == 'DyGFormerRF':
            # 验证维度约束
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                # 自动调整 channel_embedding_dim
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"自动调整 channel_embedding_dim 从 {old_dim} 到 {args.channel_embedding_dim}，"
                    f"以满足差分注意力的维度要求"
                )
            
            dynamic_backbone = DyGFormerRF(
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
        
        elif args.model_name == 'DyGFormerRRF':
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"自动调整 channel_embedding_dim 从 {old_dim} 到 {args.channel_embedding_dim}，"
                    f"以满足差分注意力的维度要求"
                )
            
            dynamic_backbone = DyGFormerRRF(
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
        
        elif args.model_name == 'DyGFormerRFRF':
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"自动调整 channel_embedding_dim 从 {old_dim} 到 {args.channel_embedding_dim}，"
                    f"以满足差分注意力的维度要求"
                )
            
            dynamic_backbone = DyGFormerRFRF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                use_second_hop=getattr(args, 'use_second_hop', False),
                num_neighbors=getattr(args, 'num_neighbors', 20),
            )
        
        elif args.model_name == 'DyGFormerFR':
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"自动调整 channel_embedding_dim 从 {old_dim} 到 {args.channel_embedding_dim}，"
                    f"以满足差分注意力的维度要求"
                )

            dynamic_backbone = DyGFormerFR(
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
        
        elif args.model_name == 'DyGFormerPF':
            # 验证维度约束
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                # 自动调整 channel_embedding_dim
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"自动调整 channel_embedding_dim 从 {old_dim} 到 {args.channel_embedding_dim}，"
                    f"以满足差分注意力的维度要求"
                )
            
            dynamic_backbone = DyGFormerPF(
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
        
        elif args.model_name == 'DyGFormerWSDMP':
            dynamic_backbone = DyGFormerWSDMP(
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

        elif args.model_name == 'DyGFormerWSDMPC':
            dynamic_backbone = DyGFormerWSDMPC(
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

        elif args.model_name == 'DyGFormerX':
            dynamic_backbone = DyGFormerX(
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

        elif args.model_name == 'DyGFormerXR':
            dynamic_backbone = DyGFormerXR(
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

        elif args.model_name == 'DyGFormerWSDMPX':
            dynamic_backbone = DyGFormerWSDMPX(
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

        elif args.model_name == 'DyGFormerWSDMPF':
            dynamic_backbone = DyGFormerWSDMPF(
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
        
        elif args.model_name == 'DyGFormerWSDMR':
            dynamic_backbone = DyGFormerWSDMR(
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
        
        elif args.model_name == 'DyGFormerC':
            dynamic_backbone = DyGFormerC(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        elif args.model_name == 'DyGFormerCP':
            dynamic_backbone = DyGFormerCP(
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
        elif args.model_name == 'DyGFormerCR':
            dynamic_backbone = DyGFormerCR(
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
        elif args.model_name == 'DyGFormerDEG':
            dynamic_backbone = DyGFormerDEG(
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
                max_degree=getattr(args, 'max_degree', 512),
                device=args.device
            )
        elif args.model_name == 'DyGFormerPC':
            dynamic_backbone = DyGFormerPC(
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
                device=args.device,
                dp_rank=getattr(args, 'dp_rank', 8),
                alpha=getattr(args, 'alpha', 0.9),
                d_ff=getattr(args, 'd_ff', None),
                merge_size=getattr(args, 'merge_size', 1),
                momentum=getattr(args, 'momentum', 0.1)
            )
        elif args.model_name == 'DyGFormerRC':
            dynamic_backbone = DyGFormerRC(
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
                device=args.device,
                dp_rank=getattr(args, 'dp_rank', 8),
                alpha=getattr(args, 'alpha', 0.9),
                d_ff=getattr(args, 'd_ff', None),
                merge_size=getattr(args, 'merge_size', 1),
                momentum=getattr(args, 'momentum', 0.1)
            )
        
        elif args.model_name == 'DyGFormerD':
            from models.DyGFormerD import DyGFormerD
            dynamic_backbone = DyGFormerD(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length,
                                         use_lpu=getattr(args, 'use_lpu', True), use_rpe=getattr(args, 'use_rpe', True),
                                         rwpe_dim=getattr(args, 'rwpe_dim', 8),
                                         rwpe_precomputed_dir=getattr(args, 'rwpe_precomputed_dir', None),
                                         rwpe_lazy_load=getattr(args, 'rwpe_lazy_load', False),
                                         max_subgraph_size=getattr(args, 'rwpe_max_subgraph_size', 512),
                                         device=args.device)
        
        elif args.model_name == 'DyGFormerST':
            from models.DyGFormerST import DyGFormerST
            dynamic_backbone = DyGFormerST(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length,
                                         max_subgraph_size=getattr(args, 'spd_max_subgraph_size', 512),
                                         spd_precomputed_dir=getattr(args, 'spd_precomputed_dir', None),
                                         spd_lazy_load=getattr(args, 'spd_lazy_load', False),
                                         device=args.device)
        
        elif args.model_name == 'DyGFormerP':
            dynamic_backbone = DyGFormerP(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        elif args.model_name == 'DyGFormerR':
            dynamic_backbone = DyGFormerR(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        elif args.model_name == 'DyGFormerRS':
            from models.DyGFormerRS import DyGFormerRS
            dynamic_backbone = DyGFormerRS(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        elif args.model_name == 'PDyGFormer':
            # PDyGFormer 支持多尺度自适应 patching
            use_multi_scale = getattr(args, 'use_multi_scale', True)
            patch_size_list = getattr(args, 'patch_size_list', [8, 6, 4, 2])
            
            dynamic_backbone = PDyGFormer(
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
                device=args.device,
                use_multi_scale=use_multi_scale,
                patch_size_list=patch_size_list
            )
        
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
                device=args.device
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
        elif args.model_name == 'NPTSTDyG':
            dynamic_backbone = NPTSTDyG(
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
        elif args.model_name == 'NPTWSDMP':
            dynamic_backbone = NPTWSDMP(
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
        elif args.model_name == 'OCARDDyG':
            dynamic_backbone = OCARDDyG(
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
                device=args.device,
                dp_rank=getattr(args, 'dp_rank', 8),
                alpha=getattr(args, 'alpha', 0.9),
                d_ff=getattr(args, 'd_ff', None),
                merge_size=getattr(args, 'merge_size', 1),
                momentum=getattr(args, 'momentum', 0.1)
            )
        elif args.model_name == 'CCARDDyG':
            dynamic_backbone = CCARDDyG(
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
                device=args.device,
                dp_rank=getattr(args, 'dp_rank', 8),
                alpha=getattr(args, 'alpha', 0.9),
                d_ff=getattr(args, 'd_ff', None),
                merge_size=getattr(args, 'merge_size', 1),
                momentum=getattr(args, 'momentum', 0.1)
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
        elif args.model_name == 'DyGFormerGPR':
            dynamic_backbone = DyGFormerGPR(
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
                device=args.device,
                K_transformer=getattr(args, 'K_transformer', 10),
                alpha=getattr(args, 'alpha', 0.1)
            )
        elif args.model_name == 'DyGFormerHDSE':
            dynamic_backbone = DyGFormerHDSE(
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
                device=args.device,
                max_spd=getattr(args, 'max_spd', 30),
                max_dist=getattr(args, 'max_dist', 10),
                struct_embed_dim=getattr(args, 'struct_embed_dim', 16),
                max_temp_struct_nodes=getattr(args, 'max_temp_struct_nodes', 200),
            )
        elif args.model_name == 'DyGFormerSPSE':
            dynamic_backbone = DyGFormerSPSE(
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
                device=args.device,
                spse_dist_len=getattr(args, 'spse_dist_len', 2),
                spse_root_nodes_fact=getattr(args, 'spse_root_nodes_fact', 0.075),
                spse_min_root_nodes=getattr(args, 'spse_min_root_nodes', 4),
                spse_max_root_nodes=getattr(args, 'spse_max_root_nodes', 0),
                spse_max_bfs_depth=getattr(args, 'spse_max_bfs_depth', 0),
                spse_max_tries_per_depth=getattr(args, 'spse_max_tries_per_depth', 1),
                spse_use_adjacency_walk_counts=not getattr(args, 'spse_use_full_spse', False),
                spse_max_subgraph_nodes=getattr(args, 'spse_max_subgraph_nodes', 200),
                spse_precomputed_dir=getattr(args, 'spse_precomputed_dir', None),
                spse_lazy_load=(args.dataset_name == 'UNvote'),
            )
        elif args.model_name == 'DyGFormerRRWP':
            dynamic_backbone = DyGFormerRRWP(
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
                device=args.device,
                rrwp_walk_length=getattr(args, 'rrwp_walk_length', 8),
                rrwp_add_identity=getattr(args, 'rrwp_add_identity', True),
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
        elif args.model_name == 'DyGFormerCC':
            from models.DyGFormerCC import DyGFormerCC
            dynamic_backbone = DyGFormerCC(
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
                device=args.device,
                dp_rank=getattr(args, 'dp_rank', 16),
                alpha=getattr(args, 'alpha', 0.5),
                merge_size=getattr(args, 'merge_size', 1),
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        # 创建link_predictor - 根据不同模型使用不同的输入维度
        if args.model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerFR', 'DyGFormerRF', 'DyGFormerRFRF', 'DyGFormerRRF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'TIMERDyG', 'STDyGFormer', 'RLDyGFormer','LDyGFormer', 'PDyGFormer', 'DyGFormerST', 'DyGFormerRS', 'DyGFormerTW', 'DyGFormerPF', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPC', 'DyGFormerWSDMPF', 'DyGFormerWSDMR', 'DyGFormerCP', 'DyGFormerCR', 'DyGFormerDEG', 'DyGFormerPC', 'DyGFormerRC', 'DyGFormerGPR', 'DyGFormerHDSE', 'DyGFormerSPSE', 'DyGFormerRRWP', 'DyGFormerX', 'DyGFormerXR', 'DyGFormerWSDMPX', 'DyGFormerCC']:
            # DyGFormerF、DyGFormerFF、DyGFormerST 等输出是 node_feat_dim，与其他 DyGFormer 系列一致
            input_dim1 = dynamic_backbone.node_feat_dim
            input_dim2 = dynamic_backbone.node_feat_dim
            link_predictor = MergeLayer(
                input_dim1=input_dim1,
                input_dim2=input_dim2,
                hidden_dim=128,
                output_dim=1
             )
        elif args.model_name in ['DeformableDyG', 'PathDyG', 'CARDDyG', 'OCARDDyG', 'CCARDDyG']:
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
        elif args.model_name in ['NPTDyG', 'NPTSTDyG']:
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
             )
        elif args.model_name in ['DIFFDyG', 'STDIFFDyG', 'RLDIFFDyG','LDIFFDyG','RLSDIFFDyG','EMADIFFDyG','SRLDIFFDyG','HRLDIFFDyG']:
            # DIFFDyG 系列模型的 output_layer 将特征映射到 node_feat_dim
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
                    f'{get_parameter_sizes(model) * 4 / 1024:.4f} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024:.4f} MB.')

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

        loss_func = nn.BCEWithLogitsLoss()

        # 评估最佳模型
        logger.info(f'在 {args.dataset_name} 数据集上评估模型性能...')

        # 评估测试集性能
        eval_sampler = full_neighbor_finder if args.model_name in ['STDyGFormer', 'STDIFFDyG'] else full_neighbor_sampler

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
