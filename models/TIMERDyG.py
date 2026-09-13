import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from typing import Optional, Tuple

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TIMERDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu'):
        """
        TIMERDyG model.
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
        """
        super(TIMERDyG, self).__init__()

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

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4

        self.transformers = nn.ModuleList([
            TimerDecoderLayer(
                hidden_size=self.num_channels * self.channel_embedding_dim,
                num_heads=self.num_heads,
                intermediate_size=4 * self.num_channels * self.channel_embedding_dim,
                attention_dropout=self.dropout,
                hidden_act='gelu',
                max_position_embeddings=self.max_input_sequence_length,
                layer_idx=i
            ) for i in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
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
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features, src_sequence_mask = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features, dst_sequence_mask = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_nodes_neighbor_node_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_nodes_edge_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_nodes_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_nodes_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_nodes_neighbor_node_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_nodes_edge_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_nodes_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_nodes_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_nodes_neighbor_node_features)
        src_num_patches = src_nodes_neighbor_node_features.shape[1]
        dst_num_patches = dst_nodes_neighbor_node_features.shape[1]

        # 使用 get_patches 返回的 patch mask
        # src_sequence_mask 和 dst_sequence_mask 已经在 get_patches 中生成

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        nodes_neighbor_node_features = torch.cat([src_nodes_neighbor_node_features, dst_nodes_neighbor_node_features], dim=1)
        nodes_edge_features = torch.cat([src_nodes_edge_features, dst_nodes_edge_features], dim=1)
        nodes_time_features = torch.cat([src_nodes_time_features, dst_nodes_time_features], dim=1)
        nodes_co_occurrence_features = torch.cat([src_nodes_co_occurrence_features, dst_nodes_co_occurrence_features], dim=1)

        # 合并mask
        sequence_mask = torch.cat([src_sequence_mask, dst_sequence_mask], dim=1)  # (batch, total_num_patches)

        sequence_data = [nodes_neighbor_node_features, nodes_edge_features,
                        nodes_time_features, nodes_co_occurrence_features]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        sequence_data = torch.stack(sequence_data, dim=2)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        sequence_data = sequence_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # Transformer编码部分
        for transformer in self.transformers:
            sequence_data = transformer(sequence_data, attention_mask=sequence_mask)

        # src_sequence_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_sequence_data = sequence_data[:, : src_num_patches, :]
        # dst_sequence_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_sequence_data = sequence_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # src_sequence_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_sequence_data = torch.mean(src_sequence_data, dim=1)
        # dst_sequence_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_sequence_data = torch.mean(dst_sequence_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_sequence_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_sequence_data)

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
        # 确保序列长度是 patch_size 的倍数
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
        # 保证 node_interact_times 和 padded_nodes_neighbor_times 都是 numpy
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times_np = node_interact_times.detach().cpu().numpy()
        else:
            node_interact_times_np = node_interact_times
        if isinstance(padded_nodes_neighbor_times, torch.Tensor):
            padded_nodes_neighbor_times_np = padded_nodes_neighbor_times.detach().cpu().numpy()
        else:
            padded_nodes_neighbor_times_np = padded_nodes_neighbor_times

        # 计算时间差
        time_deltas = node_interact_times_np[:, np.newaxis] - padded_nodes_neighbor_times_np
        # 转成 tensor 并放到目标 device
        time_deltas_tensor = torch.from_numpy(time_deltas).float().to(self.device)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=time_deltas_tensor)

        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]

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
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        # 生成patch mask: 只要patch中有非零特征就视为有效
        patch_sum = torch.sum(torch.abs(patches_nodes_neighbor_node_raw_features), dim=2)  # (batch, num_patches)
        patch_mask = (patch_sum > 0).float()  # (batch, num_patches)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features, patch_mask

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

    def forward(self, batch):
        """
        batch: dict，包含 'src_node_ids', 'dst_node_ids', 'node_interact_times'
        """
        src_node_ids = batch['src_node_ids']
        dst_node_ids = batch['dst_node_ids']
        node_interact_times = batch['node_interact_times']
        return self.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times
        )


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


class TimerMoeRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class TimerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout=0.1, max_position_embeddings=10000, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask, dropout_p=self.attention_dropout)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class TimerMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = 'gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = getattr(F, hidden_act) if hasattr(F, hidden_act) else F.gelu
    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class TimerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attention_dropout=0.1, hidden_act='gelu', max_position_embeddings=10000, layer_idx=0):
        super().__init__()
        self.self_attn = TimerAttention(hidden_size, num_heads, attention_dropout, max_position_embeddings, layer_idx)
        self.ffn_layer = TimerMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)

    def build_causal_mask(self, attention_mask):
        batch_size, seq_len = attention_mask.shape
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)
        mask = (1.0 - causal_mask) * -1e9
        return mask

    def forward(self, hidden_states: torch.Tensor, position_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states
        attn_mask = None
        if attention_mask is not None:
            attn_mask = self.build_causal_mask(attention_mask)
        hidden_states = self.self_attn(hidden_states, position_ids=position_ids, attention_mask=attn_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.norm1(hidden_states)
        residual = hidden_states
        hidden_states = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm2(hidden_states)
        return hidden_states
