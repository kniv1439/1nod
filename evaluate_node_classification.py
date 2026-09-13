import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch.nn as nn
import torch
import ast
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.RepeatMixer import RepeatMixer
from models.DyGFormer import DyGFormer
from models.DyGFormerTW import DyGFormerTW
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
from models.DIFFDyG import DIFFDyG  # Import DIFFDyG model
from models.PathDyG import PathDyG  # Import PathDyG model
from models.NPTDyG import NPTDyG  # Import NPTDyG model
from models.NPTSTDyG import NPTSTDyG
from models.PDyGFormer import PDyGFormer
from models.DyGFormerGPR import DyGFormerGPR
from models.DyGFormerHDSE import DyGFormerHDSE
from models.DyGFormerX import DyGFormerX
from models.DyGFormerXR import DyGFormerXR
from models.DyGFormerWSDMPX import DyGFormerWSDMPX
from models.DyGFormerCC import DyGFormerCC
from models.modules import MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, get_directed_neighbor_sampler, get_edge_neighbor_sampler
from evaluate_models_utils import evaluate_model_node_classification
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args, build_run_name_suffix
from models.TIMERDyG import TIMERDyG


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args()

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
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    start_seed = getattr(args, 'start_seed', 0)
    for run in range(start_seed, start_seed + args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'node_classification_{args.model_name}_{args.node_cls_train_mode}_seed{args.seed}'
        args.save_result_name = f'evaluate_node_classification_{args.model_name}_{args.node_cls_train_mode}_seed{args.seed}'
        suffix = build_run_name_suffix(args, include_batch=True)
        if suffix:
            args.load_model_name += '_' + suffix
            args.save_result_name += '_' + suffix

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model（分类器统一使用 models.modules.MLPClassifier，按模型设置 input_dim）
        classifier_input_dim = node_raw_features.shape[1]
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
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
                device=args.device
            )
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
        
        elif args.model_name == 'DyGFormerF':
            # 确保维度满足差分注意力的要求
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
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
            logger.info(f"DyGFormerF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
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
                device=args.device
            )
            logger.info(f"DyGFormerFF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
        elif args.model_name == 'DyGFormerRF':
            # 确保维度满足差分注意力的要求
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
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
                device=args.device
            )
            logger.info(f"DyGFormerRF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
        
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
                device=args.device,
                use_second_hop=getattr(args, 'use_second_hop', False),
                num_neighbors=getattr(args, 'num_neighbors', 20),
            )
            logger.info(f"DyGFormerRFRF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}")
        
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
                device=args.device
            )
            logger.info(f"DyGFormerFR配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
        
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
                device=args.device
            )
            logger.info(f"DyGFormerPF配置: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
        
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
                device=args.device,
            )
        
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
        
        elif args.model_name == 'DIFFDyG':
            dynamic_backbone = DIFFDyG(node_raw_features=node_raw_features,edge_raw_features=edge_raw_features,neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim,channel_embedding_dim=args.channel_embedding_dim,
                                       num_layers=args.num_layers,num_heads=args.num_heads,dropout=args.dropout,
                                       max_input_sequence_length=args.max_input_sequence_length,device=args.device)
        elif args.model_name == 'NPTDyG':
            dynamic_backbone = NPTDyG(
                node_raw_features=node_raw_features,edge_raw_features=edge_raw_features,neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,channel_embedding_dim=args.channel_embedding_dim,num_layers=args.num_layers,
                num_heads=args.num_heads,dropout=args.dropout,max_input_sequence_length=args.max_input_sequence_length,device=args.device)
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
        elif args.model_name == 'PathDyG':
            classifier_input_dim = 4 * args.channel_embedding_dim  # 与训练脚本一致
            dynamic_backbone = PathDyG(node_raw_features=node_raw_features,edge_raw_features=edge_raw_features,neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim,channel_embedding_dim=args.channel_embedding_dim,num_layers=args.num_layers,
                                       num_heads=args.num_heads,dropout=args.dropout,max_input_sequence_length=args.max_input_sequence_length, device=args.device
    )
        elif args.model_name == 'TIMERDyG':
            dynamic_backbone = TIMERDyG(
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
                neighbor_finder=full_neighbor_sampler,
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
            classifier_input_dim = dynamic_backbone.node_feat_dim
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
            classifier_input_dim = dynamic_backbone.node_feat_dim
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
                device=args.device,
                dp_rank=getattr(args, 'dp_rank', 16),
                alpha=getattr(args, 'alpha', 0.5),
                merge_size=getattr(args, 'merge_size', 1),
            )
            classifier_input_dim = dynamic_backbone.node_feat_dim
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        
        # 统一使用 models.modules 中的 MLPClassifier（与训练脚本一致）
        node_classifier = MLPClassifier(input_dim=classifier_input_dim, dropout=args.dropout)
        model = nn.Sequential(dynamic_backbone, node_classifier)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024:.4f} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024:.4f} MB.')

        # load the saved model
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                       save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')

        model = convert_to_gpu(model, device=args.device)

        # put the node raw messages of memory-based models on device
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        loss_func = nn.BCELoss()

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_total_loss, val_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=val_idx_data_loader,
                                                                             evaluate_data=val_data,
                                                                             loss_func=loss_func,
                                                                             num_neighbors=args.num_neighbors,
                                                                             time_gap=args.time_gap,
                                                                             edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None)

        test_total_loss, test_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap,
                                                                           edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                val_metric = val_metrics[metric_name]
                logger.info(f'validate {metric_name}, {val_metric:.4f}')
                val_metric_dict[metric_name] = val_metric

        logger.info(f'test loss: {test_total_loss:.4f}')
        for metric_name in test_metrics.keys():
            test_metric = test_metrics[metric_name]
            logger.info(f'test {metric_name}, {test_metric:.4f}')
            test_metric_dict[metric_name] = test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < start_seed + args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save node classification results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
