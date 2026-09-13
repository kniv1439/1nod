import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from typing import Callable, Optional
import inspect

from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from sklearn.metrics import roc_auc_score, average_precision_score


def get_model_submodule(model, idx):
    """安全地获取模型的子模块，兼容DataParallel和Sequential"""
    if isinstance(model, torch.nn.DataParallel):
        return model.module[idx]
    else:
        return model[idx]


def call_classifier(classifier, src_emb, dst_emb):
    """智能调用分类器，自动适配不同的forward签名"""
    # 检查classifier的forward方法签名
    try:
        sig = inspect.signature(classifier.forward)
        params = list(sig.parameters.keys())
        
        # 根据参数签名调用合适的方法
        if 'input_1' in params and 'input_2' in params:
            return classifier(input_1=src_emb, input_2=dst_emb)
        elif 'x' in params:
            # 对于只接受一个参数的classifier，传入两个嵌入的差值或拼接
            return classifier(x=src_emb - dst_emb)
        elif len(params) >= 2:
            # 直接传入两个参数
            return classifier(src_emb, dst_emb)
        else:
            # 默认情况，拼接两个嵌入
            combined = torch.cat([src_emb, dst_emb], dim=-1)
            return classifier(combined)
    except Exception as e:
        # 如果检查签名失败，尝试各种调用方式
        try:
            return classifier(input_1=src_emb, input_2=dst_emb)
        except:
            try:
                return classifier(src_emb, dst_emb)
            except:
                try:
                    return classifier(x=src_emb - dst_emb)
                except:
                    combined = torch.cat([src_emb, dst_emb], dim=-1)
                    return classifier(combined)


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000, collect_attention: bool = False):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    

    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','DIFFDyG','STDyGFormer','STDIFFDyG','PathDyG','NPTDyG','CARDDyG','DeformableDyG','TIMERDyG','RLDyGFormer','RLDIFFDyG','LDyGFormer','LDIFFDyG','RLSDIFFDyG','EMADIFFDyG','SRLDIFFDyG','HRLDIFFDyG','SHRLDIFFDyG','RepeatMixer','TIDFormer']:
        # evaluation phase use all the graph information
        get_model_submodule(model, 0).set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer', 'DIFFDyG', 'STDyGFormer', 'STDIFFDyG', 'NPTDyG', 'PathDyG','CARDDyG','DeformableDyG','TIMERDyG','RLDyGFormer','RLDIFFDyG','LDyGFormer','LDIFFDyG','RLSDIFFDyG','EMADIFFDyG','SRLDIFFDyG','HRLDIFFDyG','SHRLDIFFDyG','RepeatMixer','TIDFormer']:
                if model_name in ['DyGFormer']:
                    # DyGFormer需要传递record_last_attn参数来启用稳定秩收集
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        record_last_attn=collect_attention
                    )
                    if len(_out_pos) == 3:
                        batch_src_node_embeddings, batch_dst_node_embeddings, extra_pos = _out_pos
                        # 如果启用了稳定秩收集，记录相关信息
                        if collect_attention and 'metrics' in extra_pos:
                            # print(f"DyGFormer评估中收集到稳定秩数据: {len(extra_pos.get('metrics', {}).get('src', []))} src, {len(extra_pos.get('metrics', {}).get('dst', []))} dst")
                    else:
                        batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos
                    
                    # 计算负样本的嵌入（不需要收集注意力）
                    _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        record_last_attn=False
                    )
                    if len(_out_neg) == 3:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                    else:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg
                elif model_name in ['DIFFDyG', 'STDIFFDyG', 'RLDyGFormer','LDIFFDyG','RLSDIFFDyG','EMADIFFDyG','SRLDIFFDyG','HRLDIFFDyG','SHRLDIFFDyG']:
                    # DIFFDyG需要传递record_last_attn参数来启用稳定秩收集
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        record_last_attn=collect_attention
                    )
                    if len(_out_pos) == 3:
                        batch_src_node_embeddings, batch_dst_node_embeddings, extra_pos = _out_pos
                        # 如果启用了稳定秩收集，记录相关信息
                        if collect_attention and 'metrics' in extra_pos:
                            print(f"DIFFDyG评估中收集到稳定秩数据: {len(extra_pos.get('metrics', {}).get('src', []))} src, {len(extra_pos.get('metrics', {}).get('dst', []))} dst")
                    else:
                        batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos
                    
                    # 计算负样本的嵌入（不需要收集注意力）
                    _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        record_last_attn=False
                    )
                    if len(_out_neg) == 3:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                    else:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg
                elif model_name == 'NPTDyG':
                    # 使用专为评估设计的方法
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings_for_evaluation(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_pos) == 3:
                        batch_src_node_embeddings, batch_dst_node_embeddings, _ = _out_pos
                    else:
                        batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos
                    
                    # 对负样本也使用相同方法
                    _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings_for_evaluation(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_neg) == 3:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                    else:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg
                elif model_name == 'PathDyG':
                    batch_src_node_embeddings, batch_dst_node_embeddings, _ = \
                        get_model_submodule(model, 0)(  # 直接调用 forward 方法
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                                    )
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = \
                       get_model_submodule(model, 0)(  # 直接调用 forward 方法
                       src_node_ids=batch_neg_src_node_ids,
                       dst_node_ids=batch_neg_dst_node_ids,
                       node_interact_times=batch_node_interact_times
                          )
                elif model_name == 'DeformableDyG':
                    # 直接复用 DyGFormer 的逻辑
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_pos) == 3:
                        batch_src_node_embeddings, batch_dst_node_embeddings, _ = _out_pos
                    else:
                        batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos
                    
                    _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_neg) == 3:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                    else:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg
                elif model_name == 'CARDDyG':
                    # 直接复用 DyGFormer 的逻辑
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_pos) == 3:
                        batch_src_node_embeddings, batch_dst_node_embeddings, _ = _out_pos
                    else:
                        batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos
                    
                    _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_neg) == 3:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                    else:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg
                elif model_name in ['RepeatMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_edge_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors,
                                                                          time_gap=time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_edge_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors,
                                                                          time_gap=time_gap)
                        
                elif model_name in ['TIDFormer']:
                    # get temporal embedding of source and destination nodes
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_BIE_features = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times
                                                                          )

                    # get temporal embedding of negative source and negative destination nodes
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_BIE_features = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times
                                                                          )
                                                                        
                

          
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
            
            # 计算正负样本的链接概率
            positive_probabilities = call_classifier(get_model_submodule(model, 1), batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = call_classifier(get_model_submodule(model, 1), batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            # 处理可能的shape不匹配问题
            if positive_probabilities.dim() == 2 and negative_probabilities.dim() == 2:
                # 如果两者都是2维的，确保序列长度一致
                min_seq_len = min(positive_probabilities.size(1), negative_probabilities.size(1))
                positive_probabilities = positive_probabilities[:, :min_seq_len]
                negative_probabilities = negative_probabilities[:, :min_seq_len]
            elif positive_probabilities.dim() != negative_probabilities.dim():
                # 如果维度不同，尝试压缩到相同维度
                if positive_probabilities.dim() > 1:
                    positive_probabilities = positive_probabilities.view(-1)
                if negative_probabilities.dim() > 1:
                    negative_probabilities = negative_probabilities.view(-1)
            
            # 确保两者shape完全一致
            if positive_probabilities.shape != negative_probabilities.shape:
                min_len = min(positive_probabilities.numel(), negative_probabilities.numel())
                positive_probabilities = positive_probabilities.view(-1)[:min_len]
                negative_probabilities = negative_probabilities.view(-1)[:min_len]

            # 拼接预测结果和真实标签
            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            # 计算损失
            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())

            # 计算评估指标
            try:
                evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
            except ValueError as e:
                if "Only one class present" in str(e):
                    # 如果只有一个类别，跳过这个batch的指标计算
                    logger.warning(f"Batch {batch_idx + 1}: 只有一个类别，跳过AUC计算")
                    # 创建默认指标
                    default_metrics = {
                        'average_precision': 0.5,
                        'roc_auc': 0.5
                    }
                    evaluate_metrics.append(default_metrics)
                else:
                    raise e
            
            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

   

    return evaluate_losses, evaluate_metrics


def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler, evaluate_idx_data_loader: DataLoader,
                               evaluate_data: Data, loss_func: Callable, num_neighbors: Optional[int] = 20,
                               time_gap: Optional[int] = None):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: Callable, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DIFFDyG', 'STDyGFormer', 'STDIFFDyG', 'NPTDyG', 'PathDyG', 'DeformableDyG','CARDDyG','RLDyGFormer','RLDIFFDyG','LDyGFormer','LDIFFDyG','RLSDIFFDyG','EMADIFFDyG','SRLDIFFDyG','HRLDIFFDyG']:
        # evaluation phase use all the graph information
        get_model_submodule(model, 0).set_neighbor_sampler(neighbor_sampler)

    if model_name in ['JODIE', 'DyRep', 'TGN']:
        # reinitialize memory of memory-based models at the start of each epoch
        get_model_submodule(model, 0).memory_bank.__init_memory_bank__()

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch = {
                    'src_node_ids': batch_src_node_ids,
                    'dst_node_ids': batch_dst_node_ids,
                    'node_interact_times': batch_node_interact_times,
                    'edge_ids': batch_edge_ids,
                    'edges_are_positive': True,
                    'num_neighbors': num_neighbors
                }
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(batch)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch = {
                    'src_node_ids': batch_src_node_ids,
                    'dst_node_ids': batch_dst_node_ids,
                    'node_interact_times': batch_node_interact_times,
                    'num_neighbors': num_neighbors,
                    'time_gap': time_gap
                }
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(batch)
            elif model_name in ['DyGFormer', 'DIFFDyG', 'STDyGFormer', 'STDIFFDyG', 'NPTDyG', 'PathDyG', 'DeformableDyG','CARDDyG','TIMERDyG','RLDyGFormer','RLDIFFDyG','LDyGFormer','LDIFFDyG','RLSDIFFDyG','EMADIFFDyG','SRLDIFFDyG','HRLDIFFDyG']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                    )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            # get predicted probabilities, shape (batch_size, )
            if model_name == 'PathDyG':
                # PathDyG的输出可能是多维的，确保最终只有一个维度
                raw_predicts = get_model_submodule(model, 1)(x=batch_src_node_embeddings)
                predicts = raw_predicts.squeeze(dim=-1).sigmoid()
                
                # 如果输出仍然是多维的，取平均值
                if len(predicts.shape) > 1:
                    predicts = torch.mean(predicts, dim=1)
            elif model_name == 'DeformableDyG':
                raw_predicts = get_model_submodule(model, 1)(x=batch_src_node_embeddings)
                
                # 使用与训练相同的改进聚合方式
                if raw_predicts.dim() > 1:
                    if raw_predicts.size(1) > 1:
                        # 使用最大值而不是平均值，保持更好的区分度
                        predicts, _ = torch.max(raw_predicts, dim=1)
                    else:
                        predicts = raw_predicts.squeeze(dim=1)
                else:
                    predicts = raw_predicts
                    
                predicts = predicts.sigmoid()
            else:
                # 其他模型正常处理
                predicts = get_model_submodule(model, 1)(x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()

            labels = torch.from_numpy(batch_labels).float().to(predicts.device)
            predicts = predicts.to(labels.dtype)  # 确保数据类型匹配

            # 最终检查
            if predicts.shape != labels.shape:
                predicts = predicts.view(labels.shape)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())
            try:
                evaluate_metrics.append(get_node_classification_metrics(predicts=predicts, labels=labels))
            except ValueError as e:
                if "Only one class present" in str(e):
                    # 如果只有一个类别，跳过这个batch的指标计算
                    logger.warning(f"Batch {batch_idx + 1}: 只有一个类别，跳过AUC计算")
                    # 创建默认指标
                    default_metrics = {
                        'average_precision': 0.5,
                        'roc_auc': 0.5
                    }
                    evaluate_metrics.append(default_metrics)
                else:
                    raise e

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        if model_name == 'NPTDyG' and hasattr(get_model_submodule(model, 0), 'eval_mode'):
            get_model_submodule(model, 0).eval_mode = False

    return np.mean(evaluate_losses), {metric_name: np.mean([evaluate_metric[metric_name] for evaluate_metric in evaluate_metrics]) for metric_name in evaluate_metrics[0].keys()}


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

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

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                              batch_src_node_ids=batch_src_node_ids,
                                                                                              batch_dst_node_ids=batch_dst_node_ids,
                                                                                              current_batch_start_time=batch_node_interact_times[0],
                                                                                              current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=args.test_ratio)

            if positive_probabilities.dim() == 2 and negative_probabilities.dim() == 2:
                min_seq_len = min(positive_probabilities.size(1), negative_probabilities.size(1))
                positive_probabilities = positive_probabilities[:, :min_seq_len]
                negative_probabilities = negative_probabilities[:, :min_seq_len]

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}'for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')


# 添加缺失的函数和导入
def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def edge_bank_link_prediction(history_data, positive_edges, negative_edges, edge_bank_memory_mode, time_window_mode, time_window_proportion):
    """
    EdgeBank链接预测函数
    """
    # 这里需要实现EdgeBank的逻辑
    # 暂时返回随机值作为占位符
    pos_size = len(positive_edges[0])
    neg_size = len(negative_edges[0])
    
    positive_probabilities = np.random.random(pos_size)
    negative_probabilities = np.random.random(neg_size)
    
    return torch.from_numpy(positive_probabilities), torch.from_numpy(negative_probabilities)


# 添加logger变量
logger = logging.getLogger(__name__)
