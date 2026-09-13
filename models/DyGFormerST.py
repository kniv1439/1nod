import os
import glob
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class DyGFormerST(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu',
                 max_subgraph_size: int = 512, spd_precomputed_dir: str = None, spd_lazy_load: bool = False):
        """
        DyGFormerST model with spatial-temporal features.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        :param max_subgraph_size: int, SPD 子图 unique 节点数上限，超过则该样本 SPD 全部置零（与预计算保持一致）
        :param spd_precomputed_dir: str, 预计算 SPD（最短路径距离）目录（edge_id 键，只覆盖正样本）。
            为 None 时完全在线计算，行为与旧版一致。
        :param spd_lazy_load: bool, 大数据集懒加载 chunk（不一次性把所有 chunk 读进内存）
        """
        super(DyGFormerST, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        self.max_subgraph_size = max_subgraph_size

        # ── 预计算 SPD（最短路径距离）缓存（可选，edge_id 键，只覆盖正样本；
        #    与 precompute_rwpe_modelfn.py / DyGFormerD 旧版 edge_id RWPE 完全同构）──
        self._spd_precomputed = None      # 全量加载模式: {edge_id: {'src': np.int16[Ls], 'dst': np.int16[Ld]}}
        self._spd_precomp_dir = None      # 懒加载模式: chunk 目录
        self._spd_chunk_size = None
        self._spd_chunk_cache = {}
        self._spd_chunk_lru = []
        self._spd_max_cached_chunks = 8
        self._spd_eid_to_row = None

        if spd_precomputed_dir is not None:
            self._init_spd_precomputed(spd_precomputed_dir, spd_lazy_load)

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        # 添加 spatial encoder
        self.spatial_pos_embedding = nn.Embedding(
            num_embeddings=512,  # 最大距离限制为 511
            embedding_dim=self.time_feat_dim
        )

        # 在 projection_layer 中添加 spatial 投影层
        self.projection_layer['spatial'] = nn.Linear(
            in_features=self.patch_size * self.time_feat_dim,
            out_features=self.channel_embedding_dim,
            bias=True
        )

        self.num_channels = 5  # 从 4 改为 5，添加 spatial 通道

        # 计算原始 attention_dim
        original_attention_dim = self.num_channels * self.channel_embedding_dim
        # 确保 attention_dim 能被 num_heads 整除
        if original_attention_dim % self.num_heads != 0:
            # 向上取整到最近的能被 num_heads 整除的值
            self.attention_dim = ((original_attention_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
            # 添加维度调整层
            self.attention_dim_adjust = nn.Linear(in_features=original_attention_dim, out_features=self.attention_dim, bias=True)
            self.attention_dim_revert = nn.Linear(in_features=self.attention_dim, out_features=original_attention_dim, bias=True)
        else:
            self.attention_dim = original_attention_dim
            self.attention_dim_adjust = None
            self.attention_dim_revert = None

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.attention_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=original_attention_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray = None):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: optional, ndarray, shape (batch_size, ); 传入真实 edge_id 才会尝试查预计算 SPD 表
            （正样本传 batch_edge_ids，负样本传 None，与 DyGFormerD 的 edge_id 模式完全一致）
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        # 计算 spatial_pos 特征：正样本（传了 edge_ids）优先查预计算表，miss 或负样本（edge_ids=None）
        # 都回退/直接调用同一个 compute_spatial_pos_dist，保证在线值和离线表值来自同一份代码。
        if self._spd_precomp_active() and edge_ids is not None:
            src_dist, dst_dist = self._load_spd_from_precomp(
                edge_ids=edge_ids,
                src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            )
        else:
            src_dist = self.compute_spatial_pos_dist(
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                node_ids=src_node_ids,
                node_interact_times=node_interact_times
            )
            dst_dist = self.compute_spatial_pos_dist(
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times
            )

        src_spatial_pos_features = self._embed_spd_dist(src_dist)
        dst_spatial_pos_features = self._embed_spd_dist(dst_dist)

        # 将 spatial 特征分 patch
        src_patches_spatial = self.get_patches_spatial(
            spatial_features=src_spatial_pos_features,
            patch_size=self.patch_size
        )

        dst_patches_spatial = self.get_patches_spatial(
            spatial_features=dst_spatial_pos_features,
            patch_size=self.patch_size
        )

        # 投影 spatial 特征
        src_patches_spatial = self.projection_layer['spatial'](src_patches_spatial)
        dst_patches_spatial = self.projection_layer['spatial'](dst_patches_spatial)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat([src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        # 在 patches_data 中添加 spatial 通道
        patches_data = [
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
            torch.cat([src_patches_spatial, dst_patches_spatial], dim=1)  # 添加 spatial
        ]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # 如果需要，调整维度以匹配 attention_dim
        if self.attention_dim_adjust is not None:
            patches_data = self.attention_dim_adjust(patches_data)

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, attention_dim)
        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        # 如果需要，恢复原始维度
        if self.attention_dim_revert is not None:
            patches_data = self.attention_dim_revert(patches_data)

        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = torch.mean(src_patches_data, dim=1)
        # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # 确保 node_interact_times 是 numpy 数组
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()
        
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def compute_spatial_pos_dist(self, padded_nodes_neighbor_ids: np.ndarray,
                                 node_ids: np.ndarray,
                                 node_interact_times: np.ndarray) -> np.ndarray:
        """
        基于 neighbor_sampler.adj_list 构建子图并计算最短路径距离（SPD）——纯计算，不做 embedding，
        不感知任何缓存。离线预计算脚本 / 在线正样本 / 在线负样本三条路径都必须调用这同一个方法，
        保证"查表值"和"在线值"永远是同一份代码的输出。
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param node_ids: ndarray, shape (batch_size,)
        :param node_interact_times: ndarray, shape (batch_size,)
        :return: np.ndarray[int16], shape (batch_size, max_seq_length)
                 -1 表示"无信号"（padding 位置 / 有效节点数 < 2 / 子图超过 max_subgraph_size）
                 0..511 表示真实最短路径距离（已 clip）
        """
        batch_size, max_seq_length = padded_nodes_neighbor_ids.shape
        dist_out = np.full((batch_size, max_seq_length), -1, dtype=np.int16)

        # 导入 algos 模块
        try:
            from utils import algos
        except ImportError:
            # 如果无法导入，全部保持"无信号"
            return dist_out

        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy()
        node_interact_times = np.asarray(node_interact_times)
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.detach().cpu().numpy()
        node_ids = np.asarray(node_ids)

        for batch_idx in range(batch_size):
            # 获取该样本的有效节点序列（去除 padding）
            node_seq = padded_nodes_neighbor_ids[batch_idx]
            valid_nodes = node_seq[node_seq != 0]  # 去除 padding

            if len(valid_nodes) < 2:
                continue

            # 获取中心节点和当前时间
            center_node = int(node_ids[batch_idx])
            current_time = node_interact_times[batch_idx]

            # 构建唯一节点集合（包含中心节点和所有邻居）
            unique_nodes = np.unique(valid_nodes)
            N = len(unique_nodes)

            # 子图太大则跳过（与 max_subgraph_size 一致，避免 Floyd-Warshall O(N^3) 爆炸）
            if N > self.max_subgraph_size:
                continue

            node_to_local = {int(node): i for i, node in enumerate(unique_nodes)}

            # 构建邻接矩阵（基于当前时间点的历史边）
            adj = np.zeros((N, N), dtype=bool)

            # 从 neighbor_sampler.adj_list 中获取边信息
            # adj_list 格式: list of list, 每个元素是 [(neighbor_id, edge_id, timestamp), ...]
            for node_id in unique_nodes:
                node_id_int = int(node_id)
                if node_id_int < len(self.neighbor_sampler.adj_list):
                    # 获取该节点在 current_time 之前的所有邻居
                    neighbors = self.neighbor_sampler.adj_list[node_id_int]
                    for neighbor_id, edge_id, timestamp in neighbors:
                        # 只考虑时间戳 <= current_time 的边
                        if timestamp <= current_time and neighbor_id in node_to_local:
                            local_node_idx = node_to_local[node_id_int]
                            local_neighbor_idx = node_to_local[neighbor_id]
                            # 无向图，双向连接
                            adj[local_node_idx, local_neighbor_idx] = True
                            adj[local_neighbor_idx, local_node_idx] = True

            # 使用 Floyd-Warshall 计算最短路径
            try:
                shortest_path_result, _ = algos.floyd_warshall(adj)
                # 处理无穷大值（不可达节点）
                shortest_path_result = np.where(
                    np.isinf(shortest_path_result),
                    1000,  # 不可达节点设为最大距离
                    shortest_path_result
                )
                shortest_path_result = shortest_path_result.astype(np.int64)
            except Exception:
                # 如果计算失败，使用简单的距离（基于序列位置）
                shortest_path_result = np.full((N, N), fill_value=1000, dtype=np.int64)
                for i in range(N):
                    shortest_path_result[i, i] = 0

            # 将最短路径距离映射回原始序列位置
            center_local_idx = node_to_local.get(center_node, 0)
            if center_local_idx >= N:
                center_local_idx = 0

            for seq_idx, node_id in enumerate(node_seq):
                if node_id == 0:  # padding，保持 -1（无信号）
                    continue
                node_id_int = int(node_id)
                if node_id_int in node_to_local:
                    local_idx = node_to_local[node_id_int]
                    if local_idx < N and center_local_idx < N:
                        # 计算到中心节点的最短路径距离
                        dist = int(shortest_path_result[center_local_idx, local_idx])
                        # 限制最大距离
                        dist = max(0, min(dist, 511))
                        dist_out[batch_idx, seq_idx] = dist

        return dist_out

    def _embed_spd_dist(self, dist: np.ndarray) -> torch.Tensor:
        """
        把 compute_spatial_pos_dist / 预计算表查到的整数距离数组，向量化过 spatial_pos_embedding。
        :param dist: np.ndarray[int16], shape (batch_size, seq_len)，-1 表示无信号（保持零向量）
        :return: Tensor, shape (batch_size, seq_len, time_feat_dim)
        """
        batch_size, seq_len = dist.shape
        out = torch.zeros(batch_size, seq_len, self.time_feat_dim, device=self.device)
        mask = dist >= 0
        if mask.any():
            idx = torch.from_numpy(dist[mask].astype(np.int64)).to(self.device)
            out[torch.from_numpy(mask).to(self.device)] = self.spatial_pos_embedding(idx)
        return out

    # ------------------------------------------------------------------
    # 预计算 SPD 缓存基础设施（edge_id 键，只覆盖正样本，与 DyGFormerD 的
    # edge_id 模式 RWPE / precompute_rwpe_modelfn.py 完全同构）
    # ------------------------------------------------------------------

    def _init_spd_precomputed(self, spd_precomputed_dir: str, spd_lazy_load: bool):
        """加载预计算 SPD 元数据 / chunk（edge_id 键）。"""
        meta_path = os.path.join(spd_precomputed_dir, 'meta.json')
        chunk_size = 5000
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta_seq = int(meta.get('max_input_sequence_length', self.max_input_sequence_length))
            meta_ps = int(meta.get('patch_size', self.patch_size))
            meta_ms = int(meta.get('max_subgraph_size', self.max_subgraph_size))
            chunk_size = int(meta.get('chunk_size', 5000))
            if meta_seq != self.max_input_sequence_length:
                raise ValueError(
                    f'SPD precompute max_input_sequence_length={meta_seq} != model={self.max_input_sequence_length}'
                )
            if meta_ps != self.patch_size:
                raise ValueError(f'SPD precompute patch_size={meta_ps} != model={self.patch_size}')
            self.max_subgraph_size = meta_ms

        if spd_lazy_load:
            self._spd_precomp_dir = spd_precomputed_dir
            self._spd_chunk_size = chunk_size
            eid_path = os.path.join(spd_precomputed_dir, 'edge_ids.npy')
            if os.path.exists(eid_path):
                edge_ids_arr = np.load(eid_path)
                self._spd_eid_to_row = {int(e): i for i, e in enumerate(edge_ids_arr.tolist())}
            print(f'[DyGFormerST] SPD edge_id lazy-load: dir={spd_precomputed_dir}, chunk_size={chunk_size}, '
                  f'keys={0 if self._spd_eid_to_row is None else len(self._spd_eid_to_row)}')
        else:
            precomp = {}
            for path in sorted(glob.glob(os.path.join(spd_precomputed_dir, 'chunk_*.pkl'))):
                with open(path, 'rb') as f:
                    precomp.update(pickle.load(f))
            self._spd_precomputed = precomp
            print(f'[DyGFormerST] Loaded {len(precomp)} precomputed SPD entries from {spd_precomputed_dir}')

    def _load_spd_chunk_dict(self, chunk_id: int):
        cache = self._spd_chunk_cache
        if chunk_id in cache:
            self._spd_chunk_lru.remove(chunk_id)
            self._spd_chunk_lru.append(chunk_id)
            return cache[chunk_id]

        if len(cache) >= self._spd_max_cached_chunks:
            evict_id = self._spd_chunk_lru.pop(0)
            del cache[evict_id]
        chunk_path = os.path.join(self._spd_precomp_dir, f'chunk_{chunk_id:05d}.pkl')
        if os.path.exists(chunk_path):
            with open(chunk_path, 'rb') as f:
                cache[chunk_id] = pickle.load(f)
        else:
            cache[chunk_id] = {}
        self._spd_chunk_lru.append(chunk_id)
        return cache[chunk_id]

    def _get_spd_entry_edge(self, eid: int):
        """按 edge_id 查预计算 SPD 条目，miss 返回 None。"""
        if self._spd_precomputed is not None:
            return self._spd_precomputed.get(eid)

        if self._spd_precomp_dir is None:
            return None

        if self._spd_eid_to_row is not None:
            row = self._spd_eid_to_row.get(eid)
            if row is None:
                return None
            chunk_id = row // self._spd_chunk_size
        else:
            chunk_id = eid // self._spd_chunk_size

        return self._load_spd_chunk_dict(chunk_id).get(eid)

    def _spd_precomp_active(self) -> bool:
        return self._spd_precomputed is not None or self._spd_precomp_dir is not None

    @staticmethod
    def _valid_seq_len(padded_node_ids_1d: np.ndarray) -> int:
        nz = np.flatnonzero(padded_node_ids_1d != 0)
        return int(nz[-1] + 1) if len(nz) > 0 else 0

    def _load_spd_from_precomp(
        self, edge_ids,
        src_node_ids, dst_node_ids, node_interact_times,
        src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids,
    ):
        """
        仅供"传了真实 edge_ids"的调用使用（正样本）。按 edge_id 查预计算表，miss 时回退调用
        compute_spatial_pos_dist —— 和负样本调用（edge_ids=None）走的是完全同一个方法。
        :return: (src_dist, dst_dist), 均为 np.ndarray[int16]
        """
        if isinstance(edge_ids, torch.Tensor):
            edge_ids = edge_ids.detach().cpu().numpy()
        edge_ids = np.asarray(edge_ids)

        if isinstance(src_node_ids, torch.Tensor):
            src_node_ids = src_node_ids.detach().cpu().numpy()
        if isinstance(dst_node_ids, torch.Tensor):
            dst_node_ids = dst_node_ids.detach().cpu().numpy()
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy()
        src_node_ids = np.asarray(src_node_ids)
        dst_node_ids = np.asarray(dst_node_ids)
        node_interact_times = np.asarray(node_interact_times)

        B = len(edge_ids)
        src_seq_len = src_padded_nodes_neighbor_ids.shape[1]
        dst_seq_len = dst_padded_nodes_neighbor_ids.shape[1]
        src_dist = np.full((B, src_seq_len), -1, dtype=np.int16)
        dst_dist = np.full((B, dst_seq_len), -1, dtype=np.int16)

        miss = []
        for b, eid in enumerate(edge_ids):
            entry = self._get_spd_entry_edge(int(eid))
            if entry is None:
                miss.append(b)
                continue
            s, d = entry['src'], entry['dst']
            ls, ld = min(len(s), src_seq_len), min(len(d), dst_seq_len)
            if ls > 0:
                src_dist[b, :ls] = s[:ls]
            if ld > 0:
                dst_dist[b, :ld] = d[:ld]

        if miss:
            idx = np.asarray(miss)
            src_dist[idx] = self.compute_spatial_pos_dist(
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids[idx],
                node_ids=src_node_ids[idx], node_interact_times=node_interact_times[idx])
            dst_dist[idx] = self.compute_spatial_pos_dist(
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids[idx],
                node_ids=dst_node_ids[idx], node_interact_times=node_interact_times[idx])

        return src_dist, dst_dist

    def get_patches_spatial(self, spatial_features: torch.Tensor, patch_size: int = 1):
        """
        将 spatial 特征分 patch
        :param spatial_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param patch_size: int
        :return: Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        """
        assert spatial_features.shape[1] % patch_size == 0
        num_patches = spatial_features.shape[1] // patch_size
        batch_size = spatial_features.shape[0]

        patches = []
        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches.append(spatial_features[:, start_idx:end_idx, :])

        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches = torch.stack(patches, dim=1).reshape(
            batch_size, num_patches, patch_size * self.time_feat_dim
        )
        return patches


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):

            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                                                                  dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs