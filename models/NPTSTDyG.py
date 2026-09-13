import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class NPTSTDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu'):
        """
        DyGFormer model with alternating row/column attention.
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
        super(NPTSTDyG, self).__init__()

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
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim,
                              out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim,
                              out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim,
                              out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(
                in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                out_features=self.channel_embedding_dim, bias=True
            )
        })

        # 空间位置编码：距离整数索引 → time_feat_dim 维可学习向量
        self.spatial_pos_embedding = nn.Embedding(
            num_embeddings=512,
            embedding_dim=self.time_feat_dim
        )

        # spatial 通道的投影层
        self.projection_layer['spatial'] = nn.Linear(
            in_features=self.patch_size * self.time_feat_dim,
            out_features=self.channel_embedding_dim,
            bias=True
        )

        self.num_channels = 5  # node, edge, time, neighbor_co_occurrence, spatial

        # 使用行列交替注意力的 Encoder，每一层内部先 Row 再 Column
        self.transformers = nn.ModuleList([
            TransformerEncoder(
                num_channels=self.num_channels,
                channel_embedding_dim=self.channel_embedding_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=self.node_feat_dim,
            bias=True
        )

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, node_interact_times=node_interact_times
            )

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids, node_interact_times=node_interact_times
            )

        # pad the sequences of first-hop neighbors for source and destination nodes
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                nodes_edge_ids_list=src_nodes_edge_ids_list,
                nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                patch_size=self.patch_size,
                max_input_sequence_length=self.max_input_sequence_length
            )

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                nodes_edge_ids_list=dst_nodes_edge_ids_list,
                nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                patch_size=self.patch_size,
                max_input_sequence_length=self.max_input_sequence_length
            )

        # neighbor co-occurrence features
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids
            )

        # get node / edge / time features
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, \
        src_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder
            )

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, \
        dst_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder
            )

        # get patches
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size
            )

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size
            )

        # projection to channel_embedding_dim
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](
            src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](
            src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](
            src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](
            src_patches_nodes_neighbor_co_occurrence_features
        )

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](
            dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](
            dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](
            dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](
            dst_patches_nodes_neighbor_co_occurrence_features
        )

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # concat src and dst along patch dimension
        patches_nodes_neighbor_node_raw_features = torch.cat(
            [src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features],
            dim=1
        )
        patches_nodes_edge_raw_features = torch.cat(
            [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features],
            dim=1
        )
        patches_nodes_neighbor_time_features = torch.cat(
            [src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features],
            dim=1
        )
        patches_nodes_neighbor_co_occurrence_features = torch.cat(
            [src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features],
            dim=1
        )

        # 计算 spatial 位置编码（最短路径距离 → 可学习向量）
        src_spatial_features = self.compute_spatial_pos(
            padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            node_ids=src_node_ids,
            node_interact_times=node_interact_times
        )  # (batch_size, src_max_seq_length, time_feat_dim)

        dst_spatial_features = self.compute_spatial_pos(
            padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times
        )  # (batch_size, dst_max_seq_length, time_feat_dim)

        # patch 化
        src_patches_spatial = self.get_patches_spatial(src_spatial_features, self.patch_size)
        dst_patches_spatial = self.get_patches_spatial(dst_spatial_features, self.patch_size)
        # (batch_size, num_patches, patch_size * time_feat_dim)

        # 投影到 channel_embedding_dim
        src_patches_spatial = self.projection_layer['spatial'](src_patches_spatial)
        dst_patches_spatial = self.projection_layer['spatial'](dst_patches_spatial)
        # (batch_size, num_patches, channel_embedding_dim)

        # 拼接 src + dst
        patches_spatial = torch.cat([src_patches_spatial, dst_patches_spatial], dim=1)
        # (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)

        # 关键：保留 channel 维度，得到 (B, L, num_channels, C)
        patches_data = [
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
            patches_spatial,
        ]
        patches_data = torch.stack(patches_data, dim=2)  # (batch_size, L, num_channels, channel_embedding_dim)

        # 行列交替注意力
        for transformer in self.transformers:
            patches_data = transformer(patches_data)  # 仍然保持 (B, L, num_channels, C)

        # 展平 channel 维度，恢复到原来 DyGFormer 的格式
        patches_data_flat = patches_data.reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.num_channels * self.channel_embedding_dim
        )

        # split back to src / dst
        src_patches_data = patches_data_flat[:, :src_num_patches, :]
        dst_patches_data = patches_data_flat[:, src_num_patches: src_num_patches + dst_num_patches, :]

        # average pooling over patches
        src_patches_data = torch.mean(src_patches_data, dim=1)  # (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # final projection to node_feat_dim
        src_node_embeddings = self.output_layer(src_patches_data)  # (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list,
                      nodes_edge_ids_list: list, nodes_neighbor_times_list: list, patch_size: int = 1,
                      max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

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

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray, padded_nodes_neighbor_times: np.ndarray,
                     time_encoder: TimeEncoder):
        """
        get node, edge and time features
        """
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()

        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device)
        )

        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor,
                    padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor,
                    padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None,
                    patch_size: int = 1):
        """
        get the sequence of patches for nodes
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features_list = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :]
            )
            patches_nodes_edge_raw_features.append(
                padded_nodes_edge_raw_features[:, start_idx: end_idx, :]
            )
            patches_nodes_neighbor_time_features.append(
                padded_nodes_neighbor_time_features[:, start_idx: end_idx, :]
            )
            patches_nodes_neighbor_co_occurrence_features_list.append(
                padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :]
            )

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        patches_nodes_neighbor_node_raw_features = torch.stack(
            patches_nodes_neighbor_node_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)

        patches_nodes_edge_raw_features = torch.stack(
            patches_nodes_edge_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)

        patches_nodes_neighbor_time_features = torch.stack(
            patches_nodes_neighbor_time_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(
            patches_nodes_neighbor_co_occurrence_features_list, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return (patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features)

    def compute_spatial_pos(self, padded_nodes_neighbor_ids: np.ndarray,
                            node_ids: np.ndarray,
                            node_interact_times: np.ndarray):
        """
        基于 neighbor_sampler.adj_list 构建子图并计算最短路径距离（SPD），
        将整数距离通过可学习 Embedding 映射为连续向量。
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param node_ids: ndarray, shape (batch_size,)
        :param node_interact_times: ndarray, shape (batch_size,)
        :return: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        """
        batch_size, max_seq_length = padded_nodes_neighbor_ids.shape
        spatial_pos_features = torch.zeros(
            batch_size, max_seq_length, self.time_feat_dim,
            device=self.device
        )

        try:
            from utils import algos
        except ImportError:
            return spatial_pos_features

        for batch_idx in range(batch_size):
            node_seq = padded_nodes_neighbor_ids[batch_idx]
            valid_nodes = node_seq[node_seq != 0]

            if len(valid_nodes) < 2:
                continue

            center_node = int(node_ids[batch_idx])
            current_time = node_interact_times[batch_idx]

            unique_nodes = np.unique(valid_nodes)
            node_to_local = {int(node): i for i, node in enumerate(unique_nodes)}
            N = len(unique_nodes)

            adj = np.zeros((N, N), dtype=bool)
            for node_id in unique_nodes:
                node_id_int = int(node_id)
                if node_id_int < len(self.neighbor_sampler.adj_list):
                    neighbors = self.neighbor_sampler.adj_list[node_id_int]
                    for neighbor_id, edge_id, timestamp in neighbors:
                        if timestamp <= current_time and neighbor_id in node_to_local:
                            local_node_idx = node_to_local[node_id_int]
                            local_neighbor_idx = node_to_local[neighbor_id]
                            adj[local_node_idx, local_neighbor_idx] = True
                            adj[local_neighbor_idx, local_node_idx] = True

            try:
                shortest_path_result, _ = algos.floyd_warshall(adj)
                shortest_path_result = np.where(
                    np.isinf(shortest_path_result), 1000, shortest_path_result
                )
                shortest_path_result = shortest_path_result.astype(np.int64)
            except Exception:
                shortest_path_result = np.full((N, N), fill_value=1000, dtype=np.int64)
                for i in range(N):
                    shortest_path_result[i, i] = 0

            center_local_idx = node_to_local.get(center_node, 0)
            if center_local_idx >= N:
                center_local_idx = 0

            for seq_idx, node_id in enumerate(node_seq):
                if node_id == 0:
                    spatial_pos_features[batch_idx, seq_idx] = 0
                else:
                    node_id_int = int(node_id)
                    if node_id_int in node_to_local:
                        local_idx = node_to_local[node_id_int]
                        if local_idx < N and center_local_idx < N:
                            dist = int(shortest_path_result[center_local_idx, local_idx])
                            dist = max(0, min(dist, 511))
                            spatial_pos_features[batch_idx, seq_idx] = \
                                self.spatial_pos_embedding(
                                    torch.tensor(dist, device=self.device)
                                )

        return spatial_pos_features

    def get_patches_spatial(self, spatial_features: torch.Tensor, patch_size: int = 1):
        """
        将 spatial 特征序列分 patch 并拼接。
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

        patches = torch.stack(patches, dim=1).reshape(
            batch_size, num_patches, patch_size * self.time_feat_dim
        )
        return patches

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        """
        Neighbor co-occurrence encoder.
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim,
                      out_features=self.neighbor_co_occurrence_feat_dim)
        )

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray,
                                dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        """
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []

        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(
                src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):

            src_unique_keys, src_inverse_indices, src_counts = np.unique(
                src_padded_node_neighbor_ids, return_inverse=True, return_counts=True
            )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(
                src_counts[src_inverse_indices]
            ).float().to(self.device)
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True
            )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                dst_counts[dst_inverse_indices]
            ).float().to(self.device)
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                src_padded_node_neighbor_ids.copy()
            ).apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            src_padded_nodes_appearances.append(
                torch.stack([src_padded_node_neighbor_counts_in_src,
                             src_padded_node_neighbor_counts_in_dst], dim=1)
            )

            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(
                dst_padded_node_neighbor_ids.copy()
            ).apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            dst_padded_nodes_appearances.append(
                torch.stack([dst_padded_node_neighbor_counts_in_src,
                             dst_padded_node_neighbor_counts_in_dst], dim=1)
            )

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes
        """
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids
        )

        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)
        ).sum(dim=2)

        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)
        ).sum(dim=2)

        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class TransformerEncoder(nn.Module):
    """
    行列交替注意力 Encoder：
    - Row-wise: 在 patch 维度上做 self-attention（和原 DyGFormer 相同）。
    - Column-wise: 在 channel 维度上做 self-attention，使不同 channel 交互。
    """

    def __init__(self, num_channels: int, channel_embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.num_channels = num_channels
        self.channel_embedding_dim = channel_embedding_dim
        self.attention_dim = num_channels * channel_embedding_dim

        # row-wise attention: over patches, embed_dim = num_channels * channel_embedding_dim
        self.row_mha = MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # column-wise attention: over channels, embed_dim = channel_embedding_dim
        # 这里默认 num_heads 也适用于 channel_embedding_dim（例如 4 | 32）
        self.col_mha = MultiheadAttention(
            embed_dim=self.channel_embedding_dim,
            num_heads=min(num_heads, self.channel_embedding_dim),
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        # row-wise FFN + LayerNorm
        self.row_linear1 = nn.Linear(self.attention_dim, 4 * self.attention_dim)
        self.row_linear2 = nn.Linear(4 * self.attention_dim, self.attention_dim)
        self.row_norm1 = nn.LayerNorm(self.attention_dim)
        self.row_norm2 = nn.LayerNorm(self.attention_dim)

        # column-wise LayerNorm（只做简单的残差 + LN，不额外 FFN，避免维度不一致）
        self.col_norm = nn.LayerNorm(self.channel_embedding_dim)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: Tensor, shape (batch_size, num_patches, num_channels, channel_embedding_dim)
        :return: same shape
        """
        B, L, D, C = inputs.shape  # D = num_channels, C = channel_embedding_dim

        # ===== 1) Row-wise attention over patches =====
        # reshape to (B, L, D*C)
        row_input = inputs.reshape(B, L, D * C)  # (batch_size, num_patches, attention_dim)
        row_hidden = self.row_norm1(row_input)
        row_attn_out, _ = self.row_mha(row_hidden, row_hidden, row_hidden)  # (B, L, attention_dim)
        row_out = row_input + self.dropout(row_attn_out)

        row_ffn_in = self.row_norm2(row_out)
        row_ffn_out = self.row_linear2(self.dropout(F.gelu(self.row_linear1(row_ffn_in))))
        row_out = row_out + self.dropout(row_ffn_out)  # (B, L, attention_dim)

        # reshape back to (B, L, D, C)
        x = row_out.reshape(B, L, D, C)

        # ===== 2) Column-wise attention over channels =====
        # merge batch and patches: (B*L, D, C)
        col_input = x.reshape(B * L, D, C)
        col_hidden = self.col_norm(col_input)
        col_attn_out, _ = self.col_mha(col_hidden, col_hidden, col_hidden)  # (B*L, D, C)
        col_out = col_input + self.dropout(col_attn_out)

        # reshape back to (B, L, D, C)
        x = col_out.reshape(B, L, D, C)

        return x
