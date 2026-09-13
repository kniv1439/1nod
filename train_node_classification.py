# -*- coding: utf-8 -*-
import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.DIFFDyG import DIFFDyG
from models.NPTDyG import NPTDyG
from models.PathDyG import PathDyG 

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.RepeatMixer import RepeatMixer
from models.DyGFormer import DyGFormer
from models.DyGFormerTW import DyGFormerTW
from models.DyGFormerST import DyGFormerST
from models.NPTSTDyG import NPTSTDyG
from models.DyGFormerD import DyGFormerD
from models.DyGFormerP import DyGFormerP
from models.DyGFormerR import DyGFormerR
from models.DyGFormerC import DyGFormerC
from models.DyGFormerCP import DyGFormerCP
from models.DyGFormerDEG import DyGFormerDEG
from models.DyGFormerCR import DyGFormerCR
from models.DyGFormerPC import DyGFormerPC
from models.DyGFormerRC import DyGFormerRC
from models.DyGFormerF import DyGFormerF
from models.DyGFormerFF import DyGFormerFF
from models.DyGFormerRF import DyGFormerRF
from models.DyGFormerRFRF import DyGFormerRFRF
from models.DyGFormerFR import DyGFormerFR
from models.DyGFormerPF import DyGFormerPF
from models.DyGFormerWSDMP import DyGFormerWSDMP
from models.NPTWSDMP import NPTWSDMP
from models.DyGFormerWSDMPC import DyGFormerWSDMPC
from models.DyGFormerWSDMPF import DyGFormerWSDMPF
from models.DyGFormerWSDMR import DyGFormerWSDMR
from models.DeformableDyG import DeformableDyG
from models.PDyGFormer import PDyGFormer
from models.modules import MergeLayer, MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, get_directed_neighbor_sampler, get_edge_neighbor_sampler
from evaluate_models_utils import evaluate_model_node_classification
from utils.metrics import get_node_classification_metrics
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args, build_run_name_suffix
from models.TIMERDyG import TIMERDyG


from models.LDyGFormer import LDyGFormer
from models.DyGFormerGPR import DyGFormerGPR
from models.DyGFormerHDSE import DyGFormerHDSE
from models.DyGFormerX import DyGFormerX
from models.DyGFormerXR import DyGFormerXR
from models.DyGFormerWSDMPX import DyGFormerWSDMPX
from models.DyGFormerCC import DyGFormerCC
import argparse
import ast

# 后续正式参数解析和主程序



def get_backbone(model):
    # 递归剥离 DataParallel/Sequential，直到拿到真正的模型
    while True:
        if hasattr(model, "module"):
            model = model.module
        elif isinstance(model, nn.Sequential):
            model = model[0]
        else:
            break
    return model

# 新增：兼容DataParallel和普通模型的子模块获取函数
def get_submodule(model, idx):
    if isinstance(model, torch.nn.DataParallel):
        return model.module[idx]
    else:
        return model[idx]


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args()

    # 在参数解析后强制设置主卡
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    full_edge_neighbor_sampler = get_edge_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                           time_scaling_factor=args.time_scaling_factor, seed=1)

    # DyGFormerDEG requires directed neighbor sampler to distinguish in-degree and out-degree
    if args.model_name == 'DyGFormerDEG':
        full_neighbor_sampler = get_directed_neighbor_sampler(data=full_data,
                                                              sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                              time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)

    # 在初始化部分只声明列表，不打印
    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        # 若显式指定了 --lp_pretrained_model_name（如 DyGFormer_seed1）则用该 checkpoint，否则默认用 seed0，后续加载时再按存在性回退
        lp_suffix = build_run_name_suffix(args, include_batch=True)
        default_lp_name = f'{args.model_name}_seed0'
        if lp_suffix:
            default_lp_name += '_' + lp_suffix
        args.load_model_name = (args.lp_pretrained_model_name if args.lp_pretrained_model_name
                                else default_lp_name)
        args.save_model_name = f'node_classification_{args.model_name}_{args.node_cls_train_mode}_seed{args.seed}'
        nc_suffix = build_run_name_suffix(args, include_batch=True)
        if nc_suffix:
            args.save_model_name += '_' + nc_suffix

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level (INFO instead of WARNING to show important info)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 改为 INFO，避免显示 DEBUG 日志
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        # After logger init
        logger.info(f"初始化指标收集列表: run={run+1}/{args.num_runs}, val_runs={len(val_metric_all_runs)}, test_runs={len(test_metric_all_runs)}")

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=device)
        elif args.model_name == 'RepeatMixer':
            dynamic_backbone = RepeatMixer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                edge_neighbor_sampler=full_edge_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_tokens=getattr(args, 'num_tokens', args.num_neighbors),
                high_order=getattr(args, 'high_order', False),
                num_layers=args.num_layers,
                token_dim_expansion_factor=getattr(args, 'token_dim_expansion_factor', 0.5),
                channel_dim_expansion_factor=getattr(args, 'channel_dim_expansion_factor', 4.0),
                dropout=args.dropout,
                device=device
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.num_channels, dropout=args.dropout)
        elif args.model_name == 'DyGFormer':
            
            dynamic_backbone = DyGFormer(
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
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
                device=device,
                max_second_hop_length=getattr(args, 'max_second_hop_length', 32),
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'DyGFormerST':
            dynamic_backbone = DyGFormerST(
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
                max_subgraph_size=getattr(args, 'spd_max_subgraph_size', 512),
                spd_precomputed_dir=getattr(args, 'spd_precomputed_dir', None),
                spd_lazy_load=getattr(args, 'spd_lazy_load', False),
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'DyGFormerC':
            dynamic_backbone = DyGFormerC(
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
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
                device=device,
                dp_rank=getattr(args, 'dp_rank', 8),
                alpha=getattr(args, 'alpha', 0.9),
                d_ff=getattr(args, 'd_ff', None),
                merge_size=getattr(args, 'merge_size', 1),
                momentum=getattr(args, 'momentum', 0.1)
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
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
                device=device,
                dp_rank=getattr(args, 'dp_rank', 8),
                alpha=getattr(args, 'alpha', 0.9),
                d_ff=getattr(args, 'd_ff', None),
                merge_size=getattr(args, 'merge_size', 1),
                momentum=getattr(args, 'momentum', 0.1)
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerF':
            # 确保维度满足差分注意力的要求
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                # 调整为满足条件的最小值
                required_divisor = 2 * args.num_heads
                # 找到满足条件的最小channel_embedding_dim
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
                device=device
            )
            logger.info(f"DyGFormerF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'DyGFormerFF':
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                    args.channel_embedding_dim += 1
                logger.info(f"调整channel_embedding_dim从{original_dim}到{args.channel_embedding_dim}以满足 attention_dim % (2*num_heads) == 0")
            if args.channel_embedding_dim % args.num_heads != 0:
                original_dim = args.channel_embedding_dim
                args.channel_embedding_dim = (args.channel_embedding_dim // args.num_heads) * args.num_heads
                if args.channel_embedding_dim == 0:
                    args.channel_embedding_dim = args.num_heads
                logger.info(f"调整channel_embedding_dim从{original_dim}到{args.channel_embedding_dim}以满足 RoPE (channel_embedding_dim % num_heads == 0)")
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
                device=device,
                max_second_hop_length=getattr(args, 'max_second_hop_length', 0),
            )
            logger.info(f"DyGFormerFF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'DyGFormerRF':
            # 确保维度满足差分注意力的要求
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                # 调整为满足条件的最小值
                required_divisor = 2 * args.num_heads
                while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                    args.channel_embedding_dim += 1
                logger.info(f"调整channel_embedding_dim从{original_dim}到{args.channel_embedding_dim}以满足差分注意力要求")
            
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
                device=device
            )
            logger.info(f"DyGFormerRF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerRFRF':
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                    args.channel_embedding_dim += 1
                logger.info(f"调整channel_embedding_dim从{original_dim}到{args.channel_embedding_dim}以满足差分注意力要求")
            
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
                device=device,
                use_second_hop=getattr(args, 'use_second_hop', False),
                num_neighbors=getattr(args, 'num_neighbors', 20),
            )
            logger.info(f"DyGFormerRFRF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerFR':
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                    args.channel_embedding_dim += 1
                logger.info(f"调整channel_embedding_dim从{original_dim}到{args.channel_embedding_dim}以满足差分注意力要求")

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
                device=device
            )
            logger.info(f"DyGFormerFR配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerPF':
            # 确保维度满足差分注意力的要求
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                    args.channel_embedding_dim += 1
                logger.info(f"调整channel_embedding_dim从{original_dim}到{args.channel_embedding_dim}以满足差分注意力要求")
            
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
                device=device
            )
            logger.info(f"DyGFormerPF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)

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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)

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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)

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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)

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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)

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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)

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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerD':
            dynamic_backbone = DyGFormerD(
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
                use_lpu=getattr(args, 'use_lpu', True),
                use_rpe=getattr(args, 'use_rpe', True),
                rwpe_dim=getattr(args, 'rwpe_dim', 8),
                rwpe_precomputed_dir=getattr(args, 'rwpe_precomputed_dir', None),
                rwpe_lazy_load=getattr(args, 'rwpe_lazy_load', False),
                max_subgraph_size=getattr(args, 'rwpe_max_subgraph_size', 512),
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerP':
            dynamic_backbone = DyGFormerP(
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerR':
            dynamic_backbone = DyGFormerR(
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
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
                device=device,
                use_multi_scale=use_multi_scale,
                patch_size_list=patch_size_list
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DIFFDyG':
            # 确保参数满足要求：num_channels * channel_embedding_dim 可以被 2 * num_heads 整除
            num_channels = 3  # 固定的通道数
            if (num_channels * args.channel_embedding_dim) % (2 * args.num_heads) != 0:
                original_channel_embedding_dim = args.channel_embedding_dim
                # 调整为满足条件的最小值
                required_divisor = 2 * args.num_heads
                remainder = (num_channels * args.channel_embedding_dim) % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // num_channels
                    if (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                        args.channel_embedding_dim += required_divisor // num_channels
                logger.info(f"调整channel_embedding_dim从{original_channel_embedding_dim}到{args.channel_embedding_dim}以满足数学要求")
            
            # 如果加载预训练模型，打印参数配置信息
            if hasattr(args, 'pretrained_model_name') and args.pretrained_model_name:
                logger.info(f"使用预训练模型配置创建DIFFDyG模型: "
                           f"num_layers={args.num_layers}, num_heads={args.num_heads}, "
                           f"channel_embedding_dim={args.channel_embedding_dim}, time_feat_dim={args.time_feat_dim}")
            
            dynamic_backbone = DIFFDyG(node_raw_features=node_raw_features,edge_raw_features=edge_raw_features,neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, num_heads=args.num_heads,
                                       num_layers=args.num_layers, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length, device=device)
            
            # 打印实际使用的channel_embedding_dim
            actual_channel_embedding_dim = dynamic_backbone.channel_embedding_dim
            if actual_channel_embedding_dim != args.channel_embedding_dim:
                logger.info(f"注意: channel_embedding_dim已从{args.channel_embedding_dim}调整为{actual_channel_embedding_dim}，以满足attention_dim必须被2*num_heads整除的条件")
            
            # 修改DIFFDyG部分的分类器创建代码
            output_dim = node_raw_features.shape[1]  # 使用原始节点特征维度
            logger.info(f"为DIFFDyG创建分类器(输入维度:{output_dim})")
            node_classifier = MLPClassifier(input_dim=output_dim, dropout=args.dropout)
        elif args.model_name == 'NPTDyG':
            dynamic_backbone = NPTDyG(
                node_raw_features=node_raw_features,edge_raw_features=edge_raw_features,neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,channel_embedding_dim=args.channel_embedding_dim,num_layers=args.num_layers,
                num_heads=args.num_heads,dropout=args.dropout,max_input_sequence_length=args.max_input_sequence_length,device=device)
            
            # 添加NPTDyG的分类器创建代码
            output_dim = node_raw_features.shape[1]  # 使用原始节点特征维度
            logger.info(f"为NPTDyG创建分类器(输入维度:{output_dim})")
            node_classifier = MLPClassifier(input_dim=output_dim, dropout=args.dropout)
        elif args.model_name == 'PathDyG':
            # 创建PathDyG模型
            dynamic_backbone = PathDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device
            )
            
            # 修复位置编码维度的问题
            if hasattr(dynamic_backbone, 'transformer_layer') and hasattr(dynamic_backbone.transformer_layer, 'W_pos'):
                logger.info(f"调整PathDyG的位置编码维度以匹配预训练模型")
                old_pos = dynamic_backbone.transformer_layer.W_pos
                dynamic_backbone.transformer_layer.W_pos = nn.Parameter(
                    torch.zeros(args.max_input_sequence_length, old_pos.size(1), device=old_pos.device)
                )
            
            # 使用 models.modules 的 MLPClassifier，正确设置输入维度
            # PathDyG的compute_src_dst_node_temporal_embeddings返回4个特征通道的拼接：
            # node, edge, time, neighbor_co_occurrence，每个都是channel_embedding_dim维度
            # 所以总维度是 4 * channel_embedding_dim
            output_dim = 4 * args.channel_embedding_dim  # 4 * channel_embedding_dim
            logger.info(f"为PathDyG创建分类器(输入维度:{output_dim})")
            node_classifier = MLPClassifier(input_dim=output_dim, dropout=args.dropout)
            
            # 不再跳过后续步骤
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
                device=device
            )
            
            # 动态检测DeformableDyG的实际输出维度
            try:
                with torch.no_grad():
                    # 使用小批量测试获取实际输出维度
                    test_src_ids = np.array([0])
                    test_dst_ids = np.array([1]) 
                    test_times = np.array([1.0])
                    test_src_emb, test_dst_emb = dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                        test_src_ids, test_dst_ids, test_times
                    )
                    actual_output_dim = test_src_emb.shape[-1]
                    logger.info(f"动态检测到DeformableDyG的实际输出维度: {actual_output_dim}")
                    logger.info(f"输出张量形状: {test_src_emb.shape}")
                    
                    # 检查输出值的分布
                    logger.info(f"输出值范围: [{test_src_emb.min():.4f}, {test_src_emb.max():.4f}]")
                    logger.info(f"输出值均值: {test_src_emb.mean():.4f}, 标准差: {test_src_emb.std():.4f}")
                    
            except Exception as e:
                logger.warning(f"无法动态检测维度: {e}, 使用默认维度")
                actual_output_dim = node_raw_features.shape[1]  # 使用节点原始特征维度作为fallback
            
            # 使用实际检测到的维度创建分类器
            logger.info(f"为DeformableDyG创建分类器(实际输入维度:{actual_output_dim})")
            node_classifier = MLPClassifier(input_dim=actual_output_dim, dropout=args.dropout)
        elif args.model_name == 'CARDDyG':
            dynamic_backbone = CARDDyG(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers,
                                       num_heads=args.num_heads, dropout=args.dropout, device=device)
            
            # 使用 DataParallel 包装模型
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                dynamic_backbone = nn.DataParallel(dynamic_backbone)

            dynamic_backbone.to(device)
            
            # 添加CARDDyG的分类器创建代码
            output_dim = node_raw_features.shape[1]  # 使用原始节点特征维度
            logger.info(f"为CARDDyG创建分类器(输入维度:{output_dim})")
            node_classifier = MLPClassifier(input_dim=output_dim, dropout=args.dropout)
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
                device=device,
                dp_rank=getattr(args, 'dp_rank', 8),
                alpha=getattr(args, 'alpha', 0.9),
                d_ff=getattr(args, 'd_ff', None),
                merge_size=getattr(args, 'merge_size', 1),
                momentum=getattr(args, 'momentum', 0.1)
            )
            
            # 使用 DataParallel 包装模型
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                dynamic_backbone = nn.DataParallel(dynamic_backbone)
            
            dynamic_backbone.to(device)
            
            output_dim = node_raw_features.shape[1]
            logger.info(f"为OCARDDyG创建分类器(输入维度:{output_dim})")
            node_classifier = MLPClassifier(input_dim=output_dim, dropout=args.dropout)
        elif args.model_name == 'TIMERDyG':
            dynamic_backbone = TIMERDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'STDyGFormer':
            # 解析num_neighbors参数
            num_neighbors = getattr(args, 'num_neighbors', [10, 5])
            if isinstance(num_neighbors, str):
                num_neighbors = ast.literal_eval(num_neighbors)
                
            dynamic_backbone = STDyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_finder=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device,
                k_hop=getattr(args, 'k_hop', 2),
                num_neighbors=num_neighbors
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'STDIFFDyG':
            # 解析num_neighbors参数
            num_neighbors = getattr(args, 'num_neighbors', [10, 5])
            if isinstance(num_neighbors, str):
                num_neighbors = ast.literal_eval(num_neighbors)
                
            dynamic_backbone = STDIFFDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_finder=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device,
                k_hop=getattr(args, 'k_hop', 2),
                num_neighbors=num_neighbors
            )
            # STDIFFDyG 使用 models.modules 的 MLPClassifier
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'LDyGFormer':
            dynamic_backbone = LDyGFormer(
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
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
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
                device=device,
                K_transformer=getattr(args, 'K_transformer', 10),
                alpha=getattr(args, 'alpha', 0.1)
            )
            # output_layer 映射到 node_feat_dim，与其他 DyGFormer 系列一致
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
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
                device=device,
                max_spd=getattr(args, 'max_spd', 30),
                max_dist=getattr(args, 'max_dist', 10),
                struct_embed_dim=getattr(args, 'struct_embed_dim', 16),
                max_temp_struct_nodes=getattr(args, 'max_temp_struct_nodes', 200),
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
        elif args.model_name == 'DyGFormerSPSE':
            from models.DyGFormerSPSE import DyGFormerSPSE
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
                device=device,
                spse_dist_len=getattr(args, 'spse_dist_len', 2),
                spse_precomputed_dir=getattr(args, 'spse_precomputed_dir', None),
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
        elif args.model_name == 'DyGFormerCC':
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
                device=device,
                dp_rank=getattr(args, 'dp_rank', 16),
                alpha=getattr(args, 'alpha', 0.5),
                merge_size=getattr(args, 'merge_size', 1),
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
        else:
            # 其他模型使用默认分类器
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        if args.node_cls_train_mode == 'random_finetune':
            if args.lp_pretrained_model_name:
                logger.warning(f"random_finetune 模式不加载 LP checkpoint，忽略 --lp_pretrained_model_name={args.lp_pretrained_model_name}")
            logger.info("使用随机初始化 backbone 进行节点分类 finetune，不加载 LP checkpoint")
            model = nn.Sequential(dynamic_backbone, node_classifier)
        else:
            if hasattr(dynamic_backbone, 'node_feat_dim'):
                lp_feat_dim = dynamic_backbone.node_feat_dim
            elif hasattr(dynamic_backbone, 'num_channels'):
                lp_feat_dim = dynamic_backbone.num_channels
            else:
                lp_feat_dim = node_raw_features.shape[1]
            # 与 train_link_prediction 一致：LP 中 MergeLayer 使用固定 hidden_dim=128，加载 checkpoint 时结构须一致
            link_predictor = MergeLayer(input_dim1=lp_feat_dim, input_dim2=lp_feat_dim,
                                        hidden_dim=128, output_dim=1)
            lp_model = nn.Sequential(dynamic_backbone, link_predictor)

            # load the saved model in the link prediction task
            # 若显式指定了 --lp_pretrained_model_name 则只用该 checkpoint（存在则用，不存在则报错）；否则按 seed 顺序自动选第一个存在的
            base_folder = f"./saved_models/{args.model_name}/{args.dataset_name}"
            load_model_name = args.load_model_name
            if args.lp_pretrained_model_name:
                pkl_path = os.path.join(base_folder, load_model_name, f"{load_model_name}.pkl")
                if os.path.isfile(pkl_path):
                    logger.info(f"使用指定的 LP 预训练 checkpoint: {pkl_path}")
                else:
                    raise FileNotFoundError(f"指定的 LP checkpoint 不存在: {pkl_path}（请检查 --lp_pretrained_model_name={args.lp_pretrained_model_name}）")
            else:
                seed_order = [1] + [s for s in range(args.num_runs) if s != 1]  # [1, 0, 2] when num_runs=3
                for s in seed_order:
                    candidate = f"{args.model_name}_seed{s}"
                    if lp_suffix:
                        candidate += '_' + lp_suffix
                    pkl_path = os.path.join(base_folder, candidate, f"{candidate}.pkl")
                    if os.path.isfile(pkl_path):
                        load_model_name = candidate
                        logger.info(f"使用 LP 预训练 checkpoint: {pkl_path}")
                        break
                else:
                    logger.warning(f"未找到 {base_folder}/{args.model_name}_seed{{0..{args.num_runs-1}}}[+suffix] 的 .pkl，仍使用 load_model_name={args.load_model_name}")
            load_model_folder = os.path.join(base_folder, load_model_name)
            lp_loader = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                      save_model_name=load_model_name, logger=logger, model_name=args.model_name)
            lp_loader.load_checkpoint(lp_model, map_location='cpu')

            # create the model for the node classification task
            model = nn.Sequential(lp_model[0], node_classifier)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024:.4f} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024:.4f} MB.')

        # === 多卡并行修正 ===
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs! DataParallel enabled.")
            model = torch.nn.DataParallel(model)
        else:
            print('Single GPU or CPU, DataParallel not used.')

        if args.node_cls_train_mode == 'classifier_only':
            for parameter in get_submodule(model, 0).parameters():
                parameter.requires_grad = False
            optimizer = create_optimizer(model=get_submodule(model, 1), optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        else:
            for parameter in get_submodule(model, 0).parameters():
                parameter.requires_grad = True
            optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        # put the node raw messages of memory-based models on device
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            for node_id, node_raw_messages in get_submodule(model, 0).memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(device), node_raw_message[1]))
                get_submodule(model, 0).memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        # 断点续训相关
        start_epoch = 0
        if hasattr(args, 'resume') and args.resume:
            checkpoint_path = os.path.join(save_model_folder, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"恢复训练：从第{start_epoch}个epoch继续")
            else:
                print("未找到checkpoint，无法恢复，将从头开始训练。")

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()

        epoch_train_losses = []
        epoch_val_losses = []

        if args.node_cls_train_mode == 'classifier_only':
            get_submodule(model, 0).eval()
        else:
            get_submodule(model, 0).train()

        for epoch in range(start_epoch, args.num_epochs):

            if args.node_cls_train_mode == 'classifier_only':
                get_submodule(model, 0).eval()
            else:
                get_submodule(model, 0).train()
            get_submodule(model, 1).train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerFR', 'DyGFormerRF', 'DyGFormerRFRF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerCP', 'DyGFormerCR', 'DyGFormerDEG', 'DyGFormerPC', 'DyGFormerPF', 'DyGFormerTW', 'DyGFormerST', 'NPTSTDyG', 'DIFFDyG', 'NPTDyG', 'PathDyG', 'DeformableDyG','CARDDyG', 'OCARDDyG', 'STDyGFormer', 'STDIFFDyG', 'LDyGFormer', 'PDyGFormer', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPC', 'DyGFormerWSDMPF', 'DyGFormerWSDMR', 'DyGFormerGPR', 'DyGFormerHDSE', 'DyGFormerSPSE', 'DyGFormerX', 'DyGFormerXR', 'DyGFormerWSDMPX']:
                # training process, set the neighbor sampler
                get_backbone(model).set_neighbor_sampler(full_neighbor_sampler)
            if args.model_name == 'RepeatMixer':
                get_backbone(model).set_edge_neighbor_sampler(full_edge_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                get_submodule(model, 0).memory_bank.__init_memory_bank__()

            # store train losses, trues and predicts
            train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
            batch_peak_allocated_mb_list = []
            batch_peak_reserved_mb_list = []
            epoch_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=100, 
                                            desc=f'Epoch {epoch + 1}/{args.num_epochs}', 
                                            leave=True, position=0, dynamic_ncols=True)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], \
                    train_data.edge_ids[train_data_indices], train_data.labels[train_data_indices]

                # 在每个batch送入模型前，确保所有输入张量都在主卡
                batch_src_node_ids = torch.LongTensor(batch_src_node_ids).to(device)
                batch_dst_node_ids = torch.LongTensor(batch_dst_node_ids).to(device)
                batch_node_interact_times = torch.FloatTensor(batch_node_interact_times).to(device)

                grad_context = torch.no_grad() if args.node_cls_train_mode == 'classifier_only' else torch.enable_grad()
                with grad_context:
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(
                                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                            )
                    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch = {
                            'src_node_ids': batch_src_node_ids,
                            'dst_node_ids': batch_dst_node_ids,
                            'node_interact_times': batch_node_interact_times,
                            'edge_ids': batch_edge_ids,
                            'edges_are_positive': True,
                            'num_neighbors': args.num_neighbors
                        }
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(batch)
                    elif args.model_name in ['GraphMixer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch = {
                            'src_node_ids': batch_src_node_ids,
                            'dst_node_ids': batch_dst_node_ids,
                            'node_interact_times': batch_node_interact_times,
                            'num_neighbors': args.num_neighbors,
                            'time_gap': args.time_gap
                        }
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(batch)
                    elif args.model_name == 'RepeatMixer':
                        # RepeatMixer returns single edge embedding (batch_size, num_channels)
                        batch_src_node_ids_np = batch_src_node_ids.cpu().numpy() if hasattr(batch_src_node_ids, 'cpu') else batch_src_node_ids
                        batch_dst_node_ids_np = batch_dst_node_ids.cpu().numpy() if hasattr(batch_dst_node_ids, 'cpu') else batch_dst_node_ids
                        batch_times_np = batch_node_interact_times.cpu().numpy() if hasattr(batch_node_interact_times, 'cpu') else batch_node_interact_times
                        batch_src_node_embeddings = get_backbone(model).compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids_np,
                            dst_node_ids=batch_dst_node_ids_np,
                            node_interact_times=batch_times_np,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap
                        )
                    elif args.model_name == 'DyGFormerD':
                        # node_t：按 (node,t) 查表（edge_ids 可忽略）；旧 edge_id 模式仍用 edge_ids
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=batch_edge_ids,
                            )
                    elif args.model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerFR', 'DyGFormerRF', 'DyGFormerRFRF', 'DyGFormerC', 'DyGFormerP', 'DyGFormerR', 'DyGFormerCP', 'DyGFormerCR', 'DyGFormerDEG', 'DyGFormerPC', 'DyGFormerPF', 'DyGFormerTW', 'DyGFormerST', 'NPTSTDyG', 'DIFFDyG', 'NPTDyG', 'PathDyG', 'DeformableDyG','CARDDyG', 'OCARDDyG','TIMERDyG', 'STDyGFormer', 'STDIFFDyG', 'LDyGFormer', 'PDyGFormer', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPC', 'DyGFormerWSDMPF', 'DyGFormerWSDMR', 'DyGFormerGPR', 'DyGFormerHDSE', 'DyGFormerSPSE', 'DyGFormerX', 'DyGFormerXR', 'DyGFormerWSDMPX']:
                        # get temporal embedding of source and destination nodes
                        # All these models return 2 values: (src_embeddings, dst_embeddings)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(
                                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                            )
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get predicted probabilities, shape (batch_size, )
                if args.model_name == 'PathDyG':
                    # PathDyG的输出可能是多维的，确保最终只有一个维度
                    raw_predicts = get_submodule(model, 1)(x=batch_src_node_embeddings)
                    predicts = raw_predicts.squeeze(dim=-1).sigmoid()
                    
                    # 如果输出仍然是多维的，取平均值
                    if len(predicts.shape) > 1:
                        predicts = torch.mean(predicts, dim=1)
                elif args.model_name == 'DeformableDyG':
                    # 完全恢复原始处理逻辑，不做任何聚合修改
                    predicts = get_submodule(model, 1)(x=batch_src_node_embeddings)
                    
                    # 只处理维度问题，不改变数值
                    if predicts.dim() > 1:
                        predicts = predicts.squeeze()
                    if predicts.dim() == 0:
                        predicts = predicts.unsqueeze(0)
                    
                    predicts = predicts.sigmoid()
                else:
                    # 其他模型正常处理
                    predicts = get_submodule(model, 1)(x=batch_src_node_embeddings)
                    # 确保输出是一维的
                    if len(predicts.shape) > 1:
                        # 无论有几个维度，都压缩成一维
                        predicts = predicts.squeeze()
                        
                    # 如果是标量张量（可能在batch_size=1时发生），确保它是一维的
                    if predicts.dim() == 0:
                        predicts = predicts.unsqueeze(0)
                    
                    predicts = predicts.sigmoid()
                    
                # 添加在损失计算前
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)
                predicts = predicts.to(labels.dtype)  # 确保数据类型匹配

                # 最终检查
                if predicts.shape != labels.shape:
                    logger.warning(f"调整形状: 预测值{predicts.shape} -> 标签{labels.shape}")
                    predicts = predicts.view(labels.shape)

                loss = loss_func(input=predicts, target=labels)

                train_total_loss += loss.item()

                train_y_trues.append(labels)
                train_y_predicts.append(predicts)

                accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
                if accumulation_steps > 1:
                    if batch_idx % accumulation_steps == 0:
                        optimizer.zero_grad()
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()
                    is_last_batch = (batch_idx + 1) == len(train_idx_data_loader)
                    if (batch_idx + 1) % accumulation_steps == 0 or is_last_batch:
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    batch_peak_allocated_mb_list.append(
                        torch.cuda.max_memory_allocated(device) / 1000 ** 2)
                    batch_peak_reserved_mb_list.append(
                        torch.cuda.max_memory_reserved(device) / 1000 ** 2)

                # 更新进度条，减少更新频率
                if batch_idx % 10 == 0 or batch_idx == len(train_idx_data_loader) - 1:
                    train_idx_data_loader_tqdm.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Batch': f'{batch_idx + 1}/{len(train_idx_data_loader)}'
                    })

            # ---- 每 epoch 结束后输出显存统计 ----
            if torch.cuda.is_available() and batch_peak_allocated_mb_list:
                torch.cuda.synchronize(device)
                avg_alloc = float(np.mean(batch_peak_allocated_mb_list))
                max_alloc = float(np.max(batch_peak_allocated_mb_list))
                avg_rsvd  = float(np.mean(batch_peak_reserved_mb_list))
                max_rsvd  = float(np.max(batch_peak_reserved_mb_list))
                train_time = time.time() - epoch_start_time
                memory_msg = (
                    f'[{args.model_name}][Epoch {epoch + 1}] train CUDA memory: '
                    f'avg_batch_peak_allocated={avg_alloc:.2f} MB, '
                    f'max_batch_peak_allocated={max_alloc:.2f} MB, '
                    f'avg_batch_peak_reserved={avg_rsvd:.2f} MB, '
                    f'max_batch_peak_reserved={max_rsvd:.2f} MB, '
                    f'train_time={train_time:.2f}s'
                )
                logger.info(memory_msg)
                print(memory_msg)
            # ---- 显存统计结束 ----

            train_total_loss /= (batch_idx + 1)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)

            # 计算训练指标
            try:
                train_metrics = get_node_classification_metrics(predicts=train_y_predicts, labels=train_y_trues)
                # 如果返回 nan，记录警告
                if np.isnan(train_metrics.get('roc_auc', 0)):
                    if torch.any(torch.isnan(train_y_predicts)) or torch.any(torch.isinf(train_y_predicts)):
                        logger.warning("训练指标计算: 预测值中包含 nan 或 inf，无法计算 ROC-AUC")
                    else:
                        unique_labels = torch.unique(train_y_trues)
                        if len(unique_labels) < 2:
                            logger.warning(f"训练指标计算: 只有一个类别（{unique_labels[0].item()}），无法计算 ROC-AUC")
                        else:
                            logger.warning("训练指标计算: ROC-AUC 计算结果为 nan")
            except Exception as e:
                logger.warning(f"训练指标计算出错: {e}")
                train_metrics = {'roc_auc': np.nan}

            # 确保即使验证过程有问题也能收集指标
            try:
                val_total_loss, val_metrics = evaluate_model_node_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None
                )

                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}')
                for metric_name in train_metrics.keys():
                    logger.info(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
                
                # 打印epoch总结信息
                print(f"\n{'='*80}")
                print(f"Epoch {epoch + 1}/{args.num_epochs} Summary:")
                print(f"{'='*80}")
                print(f"Train Loss: {train_total_loss:.4f}")
                for metric_name in train_metrics.keys():
                    print(f"Train {metric_name}: {train_metrics[metric_name]:.4f}")
                
                logger.info(f'validate loss: {val_total_loss:.4f}')
                print(f"Validation Loss: {val_total_loss:.4f}")
                for metric_name in val_metrics.keys():
                    val_metric_value = val_metrics[metric_name]
                    # 处理 nan 值
                    if np.isnan(val_metric_value):
                        logger.warning(f'validate {metric_name} 为 nan（可能是数据极度不平衡导致某些 batch 无法计算）')
                        print(f"Validation {metric_name}: nan (数据问题，无法计算)")
                    else:
                        logger.info(f'validate {metric_name}, {val_metric_value:.4f}')
                        print(f"Validation {metric_name}: {val_metric_value:.4f}")
                print(f"{'='*80}\n")
            except Exception as e:
                logger.warning(f"验证过程发生错误: {e}")
                val_metrics = {"auc": 0.5}  # 默认指标
                val_total_loss = float('inf')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = get_submodule(model, 0).memory_bank.backup_memory_bank()

                try:
                    test_total_loss, test_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                                    model=model,
                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                                                    evaluate_data=test_data,
                                                                                    loss_func=loss_func,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap,
                                                                                    edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None)

                    logger.info(f'test loss: {test_total_loss:.4f}')
                    for metric_name in test_metrics.keys():
                        logger.info(f'test {metric_name}, {test_metrics[metric_name]:.4f}')
                except Exception as e:
                    logger.info(f"测试过程发生错误，跳过本次测试")
                    test_metrics = {"auc": 0.5}  # 默认指标
                    test_total_loss = float('inf')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    get_submodule(model, 0).memory_bank.reload_memory_bank(val_backup_memory_bank)

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                metric_value = val_metrics[metric_name]
                # 如果指标是 nan，使用 0.0 作为默认值（这样不会影响 early stopping，但会记录警告）
                if np.isnan(metric_value):
                    logger.warning(f"验证指标 {metric_name} 为 nan，使用 0.0 进行 early stopping（可能影响模型选择）")
                    metric_value = 0.0
                val_metric_indicator.append((metric_name, metric_value, True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            # Persist newly computed (node,t) RWPE from misses
            if args.model_name == 'DyGFormerD':
                backbone = get_backbone(model)
                if hasattr(backbone, 'flush_rwpe_lazy_cache'):
                    backbone.flush_rwpe_lazy_cache()

            # 记录 epoch 级 loss
            #epoch_train_losses.append(train_total_loss)
            #epoch_val_losses.append(val_total_loss if val_total_loss != float('inf') else float('nan'))

            # 每个 epoch 保存/刷新 loss 曲线
            #loss_curve_folder = f"./loss_curve/{args.model_name}/{args.dataset_name}"
            #os.makedirs(loss_curve_folder, exist_ok=True)
           # loss_curve_path = os.path.join(
              #  loss_curve_folder,
              #  f"{args.save_model_name}_loss_curve.png"
           # )
          #  fig, ax = plt.subplots(figsize=(8, 5))
           # epochs_x = list(range(1, len(epoch_train_losses) + 1))
          #  ax.plot(epochs_x, epoch_train_losses, label='Train Loss', marker='o', markersize=3, linewidth=1.5)
           # ax.plot(epochs_x, epoch_val_losses,   label='Val Loss',   marker='s', markersize=3, linewidth=1.5)
          #  ax.set_xlabel('Epoch')
          #  ax.set_ylabel('BCE Loss')
           # ax.set_title(f'{args.model_name} on {args.dataset_name}\nLoss Curve (seed={args.seed})')
           # ax.legend()
            #ax.grid(True, alpha=0.3)
           # plt.tight_layout()
           # plt.savefig(loss_curve_path, dpi=150)
           # plt.close(fig)
           # logger.info(f"Loss 曲线已保存到 {loss_curve_path}")

            if early_stop:
                break

            # 每个epoch后保存断点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_model_folder, 'checkpoint.pth'))

        if args.model_name == 'DyGFormerD':
            backbone = get_backbone(model)
            if hasattr(backbone, 'flush_rwpe_lazy_cache'):
                backbone.flush_rwpe_lazy_cache()

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'计算最终性能评估')

        # 验证评估
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            try:
                val_total_loss, val_metrics = evaluate_model_node_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None
                )
                logger.info(f'验证指标: loss={val_total_loss:.4f}')
                val_metric_dict = {}
                for metric_name in val_metrics.keys():
                    val_metric = val_metrics[metric_name]
                    logger.info(f'验证 {metric_name}={val_metric:.4f}')
                    val_metric_dict[metric_name] = val_metric
            except Exception as e:
                logger.info(f"最终验证评估失败")
                val_metric_dict = {"auc": 0.5}  # 默认指标

        # 测试评估
        try:
            test_total_loss, test_metrics = evaluate_model_node_classification(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=test_idx_data_loader,
                evaluate_data=test_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None
            )
            logger.info(f'测试指标: loss={test_total_loss:.4f}')
            test_metric_dict = {}
            for metric_name in test_metrics.keys():
                test_metric = test_metrics[metric_name]
                logger.info(f'测试 {metric_name}={test_metric:.4f}')
                test_metric_dict[metric_name] = test_metric
        except Exception as e:
            logger.info(f"最终测试评估失败")
            test_metric_dict = {"auc": 0.5}  # 默认指标

        # 收集性能指标
        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # 确保指标被正确添加到列表中
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # 在循环内为当前 run 单独保存结果（与链路预测保持一致）
        run_result_dict = {}
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            run_result_dict["validation"] = {metric_name: float(val_metric_dict[metric_name])
                                             for metric_name in val_metric_dict}
        run_result_dict["test"] = {metric_name: float(test_metric_dict[metric_name])
                                   for metric_name in test_metric_dict}
        run_save_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(run_save_folder, exist_ok=True)
        run_save_path = os.path.join(run_save_folder, f"{args.save_model_name}.json")
        with open(run_save_path, 'w') as f:
            f.write(json.dumps(run_result_dict, indent=4))
        logger.info(f"Run {run} 结果已保存到 {run_save_path}")

        # 避免日志重叠
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    # 修改最终汇总部分，简化输出
    # store the average metrics at the log of the last run
    logger.info(f'模型评估指标汇总 ({args.num_runs}次运行):')

    # 检查验证指标
    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        if len(val_metric_all_runs) > 0:
            logger.info(f'验证指标:')
            for metric_name in val_metric_all_runs[0].keys():
                avg_metric = np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs])
                std_metric = np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1)
                logger.info(f'  {metric_name}: {avg_metric:.4f} ± {std_metric:.4f}')
        else:
            logger.info(f"无验证指标")

    # 检查测试指标
    if len(test_metric_all_runs) > 0:
        logger.info(f'测试指标:')
        for metric_name in test_metric_all_runs[0].keys():
            avg_metric = np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs])
            std_metric = np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1)
            logger.info(f'  {metric_name}: {avg_metric:.4f} ± {std_metric:.4f}')
    else:
        logger.info(f"无测试指标")

    # 保存汇总结果（均值 ± 标准差），文件名用 _summary 与单次 run 文件区分
    summary_dict = {}
    if args.model_name not in ['JODIE', 'DyRep', 'TGN'] and len(val_metric_all_runs) > 0:
        summary_dict["validation"] = {
            metric_name: {
                "mean": float(np.mean([r[metric_name] for r in val_metric_all_runs])),
                "std":  float(np.std( [r[metric_name] for r in val_metric_all_runs], ddof=1)),
            }
            for metric_name in val_metric_all_runs[0]
        }
    if len(test_metric_all_runs) > 0:
        summary_dict["test"] = {
            metric_name: {
                "mean": float(np.mean([r[metric_name] for r in test_metric_all_runs])),
                "std":  float(np.std( [r[metric_name] for r in test_metric_all_runs], ddof=1)),
            }
            for metric_name in test_metric_all_runs[0]
        }
    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    base_name = f"node_classification_{args.model_name}_{args.node_cls_train_mode}"
    summary_suffix = build_run_name_suffix(args, include_batch=True)
    if summary_suffix:
        base_name += '_' + summary_suffix
    save_result_path = os.path.join(save_result_folder, f"{base_name}_summary.json")
    with open(save_result_path, 'w') as file:
        file.write(json.dumps(summary_dict, indent=4))
    logger.info(f"汇总结果已保存到 {save_result_path}")

    sys.exit()
