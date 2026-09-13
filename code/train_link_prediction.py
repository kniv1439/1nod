import logging
import time
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

from models.DIFFDyG import DIFFDyG
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, create_optimizer, get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args


def get_model_submodule(model, idx):
    """安全地获取模型的子模块，兼容DataParallel和Sequential"""
    if isinstance(model, torch.nn.DataParallel):
        return model.module[idx]
    else:
        return model[idx]


def call_classifier(classifier, src_emb, dst_emb):
    """智能调用分类器，自动适配不同的forward签名"""
    try:
        return classifier(input_1=src_emb, input_2=dst_emb)
    except TypeError:
        try:
            return classifier(src_emb, dst_emb)
        except TypeError:
            combined = torch.cat([src_emb, dst_emb], dim=-1)
            return classifier(combined)


def train(rank, world_size, args):
    # 设备设置
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        args.device = torch.device(f'cuda:{rank}')
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取数据
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # 初始化采样器
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # 初始化负样本采样器
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # 获取数据加载器
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
        distributed=world_size > 1
    )
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):
        set_random_seed(seed=run)
        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
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
        if args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features, 
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=args.device,
                enable_stable_rank_collection=True,
                stable_rank_save_dir=f"results/stable_rank_training/{args.dataset_name}/DyGFormer_seed{run}"
            )
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
            )
        elif args.model_name == 'DIFFDyG':
            dynamic_backbone = DIFFDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                enable_stable_rank_collection=True,
                stable_rank_save_dir=f"results/stable_rank_training/{args.dataset_name}/DIFFDyG_seed{run}"
            )
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
            )
        else:
            raise ValueError(f"只支持 DIFFDyG 和 DyGFormer 模型!")

        # 组合模型
        model = nn.Sequential(dynamic_backbone, link_predictor)
        model = model.to(args.device)

        # 多卡设置
        if world_size > 1:
            if dist.is_initialized():
                model = DDP(model, device_ids=[rank], output_device=rank)
            else:
                model = DP(model)
            print(f"Using {world_size} GPUs! DataParallel/DistributedDataParallel enabled.")
        else:
            print('Single GPU or CPU, DataParallel not used.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()

        # 初始化谱间隙和稳定秩统计字典
        spectral_metrics = {
            'src': {'T_valid': [], 'sr_xxt': [], 'sr_xxt_centered': [], 'sr_xxt_perp': [], 'gap_norm': [], 'gap_norm_perp': [], 'S_op_norm': []},
            'dst': {'T_valid': [], 'sr_xxt': [], 'sr_xxt_centered': [], 'sr_xxt_perp': [], 'gap_norm': [], 'gap_norm_perp': [], 'S_op_norm': []}
        }

        for epoch in range(args.num_epochs):
            model.train()
            get_model_submodule(model, 0).set_neighbor_sampler(train_neighbor_sampler)

            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, 
                                            desc=f'Epoch {epoch + 1}/{args.num_epochs}', 
                                            ncols=120, 
                                            leave=True, 
                                            ascii=True)
            
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                batch_src_node_ids = torch.LongTensor(batch_src_node_ids).to(args.device)
                batch_dst_node_ids = torch.LongTensor(batch_dst_node_ids).to(args.device)
                batch_node_interact_times = torch.FloatTensor(batch_node_interact_times).to(args.device)

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # 计算正样本嵌入（启用注意力权重收集）
                if args.model_name == 'DyGFormer':
                    # DyGFormer 需要传递record_last_attn参数来启用稳定秩收集
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=args.num_neighbors,
                        record_last_attn=True
                    )
                else:
                    # DIFFDyG 等需要旋转嵌入的模型
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=args.num_neighbors,
                        record_last_attn=True
                    )
                if len(_out_pos) == 3:
                    batch_src_node_embeddings, batch_dst_node_embeddings, extra_pos = _out_pos
                else:
                    batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos
                    extra_pos = None

                # 计算负样本嵌入
                _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    num_neighbors=args.num_neighbors,
                    record_last_attn=False
                )
                if len(_out_neg) == 3:
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                else:
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg

                # 收集稳定秩和谱间隙统计（基于StableRankCollector一致的数据格式）
                if extra_pos is not None:
                    try:
                        if 'metrics' in extra_pos and extra_pos['metrics']:
                            for side in ['src', 'dst']:
                                if side in extra_pos['metrics']:
                                    for metric_data in extra_pos['metrics'][side]:
                                        # 收集稳定秩和谱间隙数据（与StableRankCollector一致的格式）
                                        if 'T_valid' in metric_data and 'stable_rank' in metric_data:
                                            spectral_metrics[side]['T_valid'].append(metric_data['T_valid'])
                                            spectral_metrics[side]['sr_xxt'].append(metric_data['stable_rank'])
                                            
                                            # 谱间隙指标（现在应该与StableRankCollector一致）
                                            gap_val = metric_data.get('gap_norm', None)
                                            if gap_val is not None:
                                                spectral_metrics[side]['gap_norm'].append(gap_val)
                                            
                                            # DIFFDyG特有的差分强度
                                            if args.model_name == 'DIFFDyG':
                                                s_op_val = metric_data.get('S_op_norm', None)
                                                if s_op_val is not None:
                                                    spectral_metrics[side]['S_op_norm'].append(s_op_val)
                    except Exception as e:
                        pass  # 忽略收集错误，不影响训练

                # 确保嵌入向量的维度一致性
                def normalize_embeddings(emb):
                    """标准化嵌入向量维度"""
                    if emb.dim() == 3:
                        # 如果是3维 [batch, seq, dim]，取平均或最后一个时间步
                        if emb.shape[1] == 1:
                            emb = emb.squeeze(1)  # [batch, 1, dim] -> [batch, dim]
                        else:
                            emb = emb.mean(dim=1)  # [batch, seq, dim] -> [batch, dim]
                    return emb
                
                # 标准化嵌入向量
                batch_src_node_embeddings = normalize_embeddings(batch_src_node_embeddings)
                batch_dst_node_embeddings = normalize_embeddings(batch_dst_node_embeddings)
                batch_neg_src_node_embeddings = normalize_embeddings(batch_neg_src_node_embeddings)
                batch_neg_dst_node_embeddings = normalize_embeddings(batch_neg_dst_node_embeddings)
                
                # 计算概率和损失
                positive_probabilities = call_classifier(get_model_submodule(model, 1), batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = call_classifier(get_model_submodule(model, 1), batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新进度条
                if batch_idx % 10 == 0 or batch_idx == len(train_idx_data_loader) - 1:
                    train_idx_data_loader_tqdm.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Batch': f'{batch_idx + 1}/{len(train_idx_data_loader)}'
                    })

            # 验证
            val_losses, val_metrics = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_neg_edge_sampler=val_neg_edge_sampler,
                evaluate_data=val_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap
            )

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=new_node_val_idx_data_loader,
                evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                evaluate_data=new_node_val_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap
            )

            # 打印epoch总结信息
            if rank == 0:
                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{args.num_epochs} Summary:")
            print(f"{'='*80}")
            print(f"Train Loss: {np.mean(train_losses):.4f}")
            for metric_name in train_metrics[0].keys():
                print(f"Train {metric_name}: {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}")
            
            # 注释掉打印DyGFormer的稳定秩指标
            # if args.model_name == 'DyGFormer':
            #     try:
            #         # 使用最后一个batch的数据计算指标
            #         if 'extra_pos' in locals() and extra_pos is not None:
            #             backbone = get_model_submodule(model, 0)
            #             if hasattr(backbone, '_compute_and_print_metrics_from_extra'):
            #                 backbone._compute_and_print_metrics_from_extra(extra_pos)
            #     except Exception as e:
            #         pass  # 静默处理错误
            
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            print(f"Validation Loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                val_metric_value = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {val_metric_value:.4f}')
                print(f"Validation {metric_name}: {val_metric_value:.4f}")
            
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            print(f"New Node Validation Loss: {np.mean(new_node_val_losses):.4f}")
            for metric_name in new_node_val_metrics[0].keys():
                new_node_val_metric_value = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {new_node_val_metric_value:.4f}')
                print(f"New Node Validation {metric_name}: {new_node_val_metric_value:.4f}")
            print(f"{'='*80}\n")

            # 显示谱间隙统计
            print(f"\n{'='*60}")
            # print(f"Epoch {epoch + 1} - {args.model_name} 谱间隙统计")
            print(f"{'='*60}")
            
            for side in ['src', 'dst']:
                if spectral_metrics[side]['T_valid']:
                    print(f"\n{side.upper()} 节点:")
                    print(f"  样本数: {len(spectral_metrics[side]['T_valid'])}")
                    
                    # 计算平均值和标准差
                    if spectral_metrics[side]['sr_xxt']:
                        avg_sr = np.mean(spectral_metrics[side]['sr_xxt'])
                        std_sr = np.std(spectral_metrics[side]['sr_xxt'])
                        # print(f"  稳定秩 (原始): {avg_sr:.4f} ± {std_sr:.4f}")
                    
                    # 新增：去中心化和去均匀投影的稳定秩
                    if spectral_metrics[side]['sr_xxt_centered']:
                        avg_sr_centered = np.mean(spectral_metrics[side]['sr_xxt_centered'])
                        std_sr_centered = np.std(spectral_metrics[side]['sr_xxt_centered'])
                        # print(f"  稳定秩 (去中心化): {avg_sr_centered:.4f} ± {std_sr_centered:.4f}")
                    
                    if spectral_metrics[side]['sr_xxt_perp']:
                        avg_sr_perp = np.mean(spectral_metrics[side]['sr_xxt_perp'])
                        std_sr_perp = np.std(spectral_metrics[side]['sr_xxt_perp'])
                        # print(f"  稳定秩 (去均匀投影): {avg_sr_perp:.4f} ± {std_sr_perp:.4f}")
                    
                    if spectral_metrics[side]['gap_norm']:
                        avg_gap = np.mean(spectral_metrics[side]['gap_norm'])
                        std_gap = np.std(spectral_metrics[side]['gap_norm'])
                        # print(f"  谱间隙 (原始): {avg_gap:.4f} ± {std_gap:.4f}")
                    
                    # 新增：差分注意力矩阵的谱间隙
                    if spectral_metrics[side]['gap_norm_perp']:
                        avg_gap_perp = np.mean(spectral_metrics[side]['gap_norm_perp'])
                        std_gap_perp = np.std(spectral_metrics[side]['gap_norm_perp'])
                        # print(f"  谱间隙 (去均匀投影): {avg_gap_perp:.4f} ± {std_gap_perp:.4f}")
                    
                    # DIFFDyG特有的差分强度
                    if args.model_name == 'DIFFDyG' and spectral_metrics[side]['S_op_norm']:
                        avg_S = np.mean(spectral_metrics[side]['S_op_norm'])
                        std_S = np.std(spectral_metrics[side]['S_op_norm'])
                        # print(f"  差分强度: {avg_S:.4f} ± {std_S:.4f}")
                    
                    # 显示每层稳定秩统计（基于新的数据格式）
                    if extra_pos is not None and 'metrics' in extra_pos and side in extra_pos['metrics']:
                        # print(f"\n  每层稳定秩统计:")
                        layer_groups = {}
                        
                        # 按层分组数据
                        for metric_data in extra_pos['metrics'][side]:
                            layer_idx = metric_data.get('layer', 0)
                            if layer_idx not in layer_groups:
                                layer_groups[layer_idx] = []
                            layer_groups[layer_idx].append(metric_data)
                        
                        # 显示每层统计
                        for layer_idx in sorted(layer_groups.keys()):
                            layer_data = layer_groups[layer_idx]
                            stable_ranks = [d['stable_rank'] for d in layer_data]
                            gaps = [d['gap_norm'] for d in layer_data if d['gap_norm'] is not None]
                            
                            avg_sr = np.mean(stable_ranks)
                            std_sr = np.std(stable_ranks)
                            # print(f"    第{layer_idx+1}层: 稳定秩={avg_sr:.4f} ± {std_sr:.4f}")
                            
                            if gaps:
                                avg_gap = np.mean(gaps)
                                std_gap = np.std(gaps)
                                # print(f"              谱间隙={avg_gap:.4f} ± {std_gap:.4f}")
                            
                            # DIFFDyG特有的差分强度
                            if args.model_name == 'DIFFDyG':
                                s_ops = [d['S_op_norm'] for d in layer_data if d.get('S_op_norm') is not None]
                                if s_ops:
                                    avg_s = np.mean(s_ops)
                                    std_s = np.std(s_ops)
                                    # print(f"              差分强度={avg_s:.4f} ± {std_s:.4f}")
                            
                            # print(f"              样本数={len(layer_data)}")
                    
                    # 按序列长度分组显示
                    T_groups = {}
                    for i, T in enumerate(spectral_metrics[side]['T_valid']):
                        T_key = f"T={T}"
                        if T_key not in T_groups:
                            T_groups[T_key] = {
                                'sr_xxt': [], 'gap_norm': [], 'S_op_norm': []
                            }
                        
                        T_groups[T_key]['sr_xxt'].append(spectral_metrics[side]['sr_xxt'][i])
                        
                        if i < len(spectral_metrics[side]['gap_norm']):
                            T_groups[T_key]['gap_norm'].append(spectral_metrics[side]['gap_norm'][i])
                        
                        if args.model_name == 'DIFFDyG' and i < len(spectral_metrics[side]['S_op_norm']):
                            T_groups[T_key]['S_op_norm'].append(spectral_metrics[side]['S_op_norm'][i])
                    
                    # 显示每个序列长度组的统计
                    for T_key, group_data in T_groups.items():
                        if group_data['sr_xxt']:
                            print(f"\n  {T_key}:")
                            
                            # 稳定秩统计
                            avg_sr = np.mean(group_data['sr_xxt'])
                            std_sr = np.std(group_data['sr_xxt'])
                            # print(f"    稳定秩: {avg_sr:.4f} ± {std_sr:.4f}")
                            
                            # 谱间隙统计
                            if group_data['gap_norm']:
                                avg_gap = np.mean(group_data['gap_norm'])
                                std_gap = np.std(group_data['gap_norm'])
                                # print(f"    谱间隙: {avg_gap:.4f} ± {std_gap:.4f}")
                            
                            # DIFFDyG特有的差分强度
                            if args.model_name == 'DIFFDyG' and group_data['S_op_norm']:
                                avg_S = np.mean(group_data['S_op_norm'])
                                std_S = np.std(group_data['S_op_norm'])
                                # print(f"    差分强度: {avg_S:.4f} ± {std_S:.4f}")
                            
                            # print(f"    样本数: {len(group_data['sr_xxt'])}")
                # else:
                #     print(f"\n{side.upper()} 节点: 暂无数据")
            
            # 分析注意力机制的特点
            # print(f"\n>>> {args.model_name} 注意力机制特点分析:")
            # if args.model_name == 'DIFFDyG':
            #     print("  - 差分注意力: 通过 A1 - λ*A2 计算，可能产生更丰富的谱结构")
            #     print("  - 预期: 稳定秩可能较低，谱间隙可能较大")
            #     print("  - 差分强度: 衡量A1和A2之间的差异程度")
            # else:  # DyGFormer
            #     print("  - 标准注意力: 使用 softmax(QK^T/√d) 计算")
            #     print("  - 预期: 稳定秩可能较高，谱间隙可能较小")
            #     print("  - 特点: 更平滑的注意力分布")
            
            print(f"{'='*60}\n")
            
            # 显示稳定秩收集器的统计（每个epoch结束后）
            try:
                backbone = get_model_submodule(model, 0)
                if hasattr(backbone, 'stable_rank_collector') and backbone.stable_rank_collector is not None:
                    analysis = backbone.get_stable_rank_analysis(min_sequence_length=5)
                    if analysis and 'error' not in analysis:
                        print(f"\n{'='*80}")
                        print(f"Epoch {epoch + 1} - {args.model_name} StableRankCollector 分析")
                        print(f"{'='*80}")
                        
                        print(f"总样本数: {analysis['total_samples']}")
                        
                        if 'rank_collapse' in analysis:
                            collapse_info = analysis['rank_collapse']
                            print(f"秩坍缩率: {collapse_info['collapse_rate']:.4f}")
                            print(f"坍缩样本数: {collapse_info['collapsed_samples']}")
                            print(f"坍缩阈值: {collapse_info['threshold']}")
                        
                        if 'sequence_analysis' in analysis and analysis['sequence_analysis']:
                            print(f"\n按序列长度分析:")
                            for seq_info in analysis['sequence_analysis'][:5]:  # 显示前5个序列长度的统计
                                # print(f"  T={seq_info['sequence_length']}: 平均稳定秩={seq_info['mean_stable_rank']:.4f} ± {seq_info['std_stable_rank']:.4f} (样本数={seq_info['sample_count']})")
                                pass
                        
                        if 'layer_analysis' in analysis and analysis['layer_analysis']:
                            print(f"\n按层数分析:")
                            for layer_info in analysis['layer_analysis']:
                                # print(f"  第{layer_info['layer_idx']+1}层: 平均稳定秩={layer_info['mean_stable_rank']:.4f} ± {layer_info['std_stable_rank']:.4f} (样本数={layer_info['sample_count']})")
                                pass
                        
                        # 与理论预期的比较
                        print(f"\n理论对比分析 (基于StableRankCollector):")
                        if args.model_name == 'DIFFDyG':
                            print("  - 差分注意力理论预期: 较低的秩坍缩率，更好的表征能力")
                            if 'rank_collapse' in analysis:
                                rate = analysis['rank_collapse']['collapse_rate']
                                if rate < 0.3:
                                    print(f"  - 当前表现: 优秀 (坍缩率={rate:.3f} < 0.3)")
                                elif rate < 0.5:
                                    print(f"  - 当前表现: 良好 (坍缩率={rate:.3f} < 0.5)")
                                else:
                                    print(f"  - 当前表现: 需改进 (坍缩率={rate:.3f} >= 0.5)")
                        elif args.model_name == 'DyGFormer':
                            print("  - 标准注意力理论预期: 可能出现较高的秩坍缩率")
                            if 'rank_collapse' in analysis:
                                rate = analysis['rank_collapse']['collapse_rate']
                                if rate < 0.5:
                                    print(f"  - 当前表现: 超出预期 (坍缩率={rate:.3f} < 0.5)")
                                elif rate < 0.7:
                                    print(f"  - 当前表现: 符合预期 (坍缩率={rate:.3f} < 0.7)")
                                else:
                                    print(f"  - 当前表现: 符合理论 (坍缩率={rate:.3f} >= 0.7)")
                        
                        print(f"{'='*80}\n")
                    else:
                        print(f"\n{args.model_name} StableRankCollector: 暂无足够数据或分析失败")
                        # 显示调试信息
                        if hasattr(backbone, 'stable_rank_collector') and backbone.stable_rank_collector is not None:
                            data_count = len(backbone.stable_rank_collector.layer_data)
                            print(f"  收集器状态: 已收集 {data_count} 个数据点")
                            if data_count == 0:
                                print("  可能原因: 模型前向传播时未正确收集数据")
                                print("  建议: 检查 enable_stable_rank_collection=True 且 record_last_attn=True")
                        else:
                            # print("  可能原因: 稳定秩收集器未正确初始化")
                        print()
            except Exception as e:
                print(f"\nStableRankCollector分析失败: {e}\n")

            # 测试
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_neg_edge_sampler=test_neg_edge_sampler,
                    evaluate_data=test_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap
                )

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                    evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                    evaluate_data=new_node_test_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap
                )

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                print(f"\n{'='*60}")
                print(f"Test Results (Epoch {epoch + 1}):")
                print(f"{'='*60}")
                print(f"Test Loss: {np.mean(test_losses):.4f}")
                for metric_name in test_metrics[0].keys():
                    test_metric_value = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                    logger.info(f'test {metric_name}, {test_metric_value:.4f}')
                    print(f"Test {metric_name}: {test_metric_value:.4f}")
                
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                print(f"New Node Test Loss: {np.mean(new_node_test_losses):.4f}")
                for metric_name in new_node_test_metrics[0].keys():
                    new_node_test_metric_value = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
                    logger.info(f'new node test {metric_name}, {new_node_test_metric_value:.4f}')
                    print(f"New Node Test {metric_name}: {new_node_test_metric_value:.4f}")
                print(f"{'='*60}\n")

            # Early stopping
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # 加载最佳模型
        early_stopping.load_checkpoint(model)

        # 最终评估
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        print(f"\n{'='*100}")
        print(f"FINAL EVALUATION RESULTS - {args.model_name} on {args.dataset_name}")
        print(f"{'='*100}")

        val_losses, val_metrics = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=val_idx_data_loader,
            evaluate_neg_edge_sampler=val_neg_edge_sampler,
            evaluate_data=val_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap
        )

        new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=new_node_val_idx_data_loader,
            evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
            evaluate_data=new_node_val_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap
        )

        test_losses, test_metrics = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_neg_edge_sampler=test_neg_edge_sampler,
            evaluate_data=test_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap
        )

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=new_node_test_idx_data_loader,
            evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
            evaluate_data=new_node_test_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap
        )

        # 存储评估指标
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        print(f"\nValidation Results:")
        print(f"Validation Loss: {np.mean(val_losses):.4f}")
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            print(f"Validation {metric_name}: {average_val_metric:.4f}")
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
        print(f"\nNew Node Validation Results:")
        print(f"New Node Validation Loss: {np.mean(new_node_val_losses):.4f}")
        for metric_name in new_node_val_metrics[0].keys():
            average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
            logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
            print(f"New Node Validation {metric_name}: {average_new_node_val_metric:.4f}")
            new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        print(f"\nTest Results:")
        print(f"Test Loss: {np.mean(test_losses):.4f}")
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            print(f"Test {metric_name}: {average_test_metric:.4f}")
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        print(f"\nNew Node Test Results:")
        print(f"New Node Test Loss: {np.mean(new_node_test_losses):.4f}")
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            print(f"New Node Test {metric_name}: {average_new_node_test_metric:.4f}")
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        print(f"\nRun {run + 1} completed in {single_run_time:.2f} seconds.")
        print(f"{'='*100}\n")

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # 避免日志重叠
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # 保存模型结果
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(json.dumps(result_json, indent=4))

        # 保存模型
        if rank == 0:
            torch.save(model.state_dict(), os.path.join(save_model_folder, 'model.pth'))
            
            # 保存StableRankCollector收集结果
            try:
                backbone = get_model_submodule(model, 0)
                if hasattr(backbone, 'stable_rank_collector') and backbone.stable_rank_collector is not None:
                    backbone.save_stable_rank_results(suffix=f"_final_{args.dataset_name}_run{run}")
                    logger.info(f"StableRankCollector收集结果已保存到: {backbone.stable_rank_collector.save_dir}")
                    
                    # 显示最终统计
                    final_analysis = backbone.get_stable_rank_analysis(min_sequence_length=5)
                    if final_analysis and 'error' not in final_analysis:
                        logger.info(f"最终稳定秩统计: 总样本数={final_analysis['total_samples']}")
                        if 'rank_collapse' in final_analysis:
                            logger.info(f"最终秩坍缩率: {final_analysis['rank_collapse']['collapse_rate']:.4f}")
            except Exception as e:
                logger.warning(f"保存StableRankCollector结果失败: {e}")

    # 显示所有运行的平均指标
    logger.info(f'metrics over {args.num_runs} runs:')
    
    print(f"\n{'='*120}")
    print(f"FINAL SUMMARY - {args.model_name} on {args.dataset_name} ({args.num_runs} runs)")
    print(f"{'='*120}")

    print(f"\nValidation Metrics:")
    print(f"{'-'*50}")
    for metric_name in val_metric_all_runs[0].keys():
        values = [val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        logger.info(f'validate {metric_name}, {values}')
        logger.info(f'average validate {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nNew Node Validation Metrics:")
    print(f"{'-'*50}")
    for metric_name in new_node_val_metric_all_runs[0].keys():
        values = [new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        logger.info(f'new node validate {metric_name}, {values}')
        logger.info(f'average new node validate {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nTest Metrics:")
    print(f"{'-'*50}")
    for metric_name in test_metric_all_runs[0].keys():
        values = [test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        logger.info(f'test {metric_name}, {values}')
        logger.info(f'average test {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nNew Node Test Metrics:")
    print(f"{'-'*50}")
    for metric_name in new_node_test_metric_all_runs[0].keys():
        values = [new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        logger.info(f'new node test {metric_name}, {values}')
        logger.info(f'average new node test {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n{'='*120}")
    print("Training completed successfully!")
    print(f"{'='*120}")


if __name__ == "__main__":
    args = get_link_prediction_args()
    world_size = 1
    train(rank=0, world_size=1, args=args)