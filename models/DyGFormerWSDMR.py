import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.rotary import apply_rotary_emb
from utils.RoPE import LlamaDynamicNTKScalingRotaryEmbedding


class DyGFormerWSDMR(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu'):
        """
        DyGFormer model.
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
        super(DyGFormerWSDMR, self).__init__()

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

        # 计算 attention 相关维度，确保支持 RoPE
        self.attention_dim = self.num_channels * self.channel_embedding_dim
        if (self.attention_dim // self.num_heads) % 2 != 0:
            raise ValueError(
                f"attention_dim ({self.attention_dim}) must be divisible by 2*num_heads ({2*self.num_heads})."
            )
        self.head_dim = self.attention_dim // self.num_heads // 2

        self._rotary_emb_cache = {}

        self.transformers = nn.ModuleList([
            TransformerEncoder(
                attention_dim=self.attention_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    # --------------------- RoPE ---------------------
    def _get_rotary_emb(self, seq_len: int, rotary_dim: int):
        key = rotary_dim
        if key not in self._rotary_emb_cache:
            self._rotary_emb_cache[key] = LlamaDynamicNTKScalingRotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=max(seq_len, self.max_input_sequence_length),
                base=10000,
                device=self.device,
                scaling_factor=1.0,
            ).to(self.device)
        return self._rotary_emb_cache[key]

    def compute_rotary_embedding(self, seq_len: int, rotary_dim: int):
        rotary_emb = self._get_rotary_emb(seq_len, rotary_dim)
        dummy = torch.zeros(1, 1, seq_len, rotary_dim, device=self.device, dtype=torch.float32)
        cos_cached, sin_cached = rotary_emb(dummy, seq_len=seq_len)
        cos_full = cos_cached.squeeze(0).squeeze(0)
        sin_full = sin_cached.squeeze(0).squeeze(0)
        if cos_full.shape[0] < seq_len:
            last_cos = cos_full[-1:]
            last_sin = sin_full[-1:]
            repeat_times = seq_len - cos_full.shape[0]
            cos_full = torch.cat([cos_full, last_cos.repeat(repeat_times, 1)], dim=0)
            sin_full = torch.cat([sin_full, last_sin.repeat(repeat_times, 1)], dim=0)
        cos = cos_full[:seq_len, : rotary_dim // 2].contiguous()
        sin = sin_full[:seq_len, : rotary_dim // 2].contiguous()
        return cos, sin

    def compute_token_time_space_scalars(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        src_padded_nodes_neighbor_ids: np.ndarray,
        src_padded_nodes_neighbor_times: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_times: np.ndarray,
    ):
        """
        对每个 token（节点）计算：时间标量 = 相对源的时间 + 相对目标的时间；
        空间标量 = 相对源的空间 + 相对目标的空间。
        返回 (B, L_src), (B, L_src), (B, L_dst), (B, L_dst)。
        """
        batch_size = src_padded_nodes_neighbor_ids.shape[0]
        src_len = src_padded_nodes_neighbor_ids.shape[1]
        dst_len = dst_padded_nodes_neighbor_ids.shape[1]

        cur_time = np.asarray(node_interact_times).reshape(-1, 1).astype(np.float32)
        eps = 1e-8

        # -------- src 序列：time_to_src, time_to_tgt --------
        time_to_src = np.zeros((batch_size, src_len), dtype=np.float32)
        time_to_src[:, 1:] = (cur_time - src_padded_nodes_neighbor_times[:, 1:]) / (cur_time + 1.0 + eps)
        time_to_src[src_padded_nodes_neighbor_ids == 0] = 0.0

        # 对 src 中每个 token 的 node_id，查在 dst 序列中的时间（若存在）
        time_to_tgt_src = np.ones((batch_size, src_len), dtype=np.float32)  # 默认远
        for b in range(batch_size):
            dst_id = dst_node_ids[b] if hasattr(dst_node_ids, '__getitem__') else dst_node_ids.flat[b]
            node_to_time = {}
            for k in range(dst_len):
                nid = dst_padded_nodes_neighbor_ids[b, k]
                if nid == 0:
                    continue
                t = (cur_time[b, 0] - dst_padded_nodes_neighbor_times[b, k]) / (cur_time[b, 0] + 1.0 + eps)
                node_to_time[nid] = t
            for j in range(src_len):
                nid = src_padded_nodes_neighbor_ids[b, j]
                if nid == 0:
                    continue
                if nid == dst_id:
                    time_to_tgt_src[b, j] = 0.0
                elif nid in node_to_time:
                    time_to_tgt_src[b, j] = node_to_time[nid]
        time_to_tgt_src[src_padded_nodes_neighbor_ids == 0] = 0.0

        src_token_time = time_to_src + time_to_tgt_src

        # -------- src 序列：space_to_src, space_to_tgt --------
        space_to_src = np.zeros((batch_size, src_len), dtype=np.float32)
        space_to_src[:, 1:] = 1.0
        space_to_src[src_padded_nodes_neighbor_ids == 0] = 0.0

        dst_set_per_batch = [set(dst_padded_nodes_neighbor_ids[b].flat) for b in range(batch_size)]
        space_to_tgt_src = np.full((batch_size, src_len), 2.0, dtype=np.float32)
        for b in range(batch_size):
            space_to_tgt_src[b, src_padded_nodes_neighbor_ids[b] == dst_node_ids[b]] = 0.0
            for j in range(src_len):
                nid = src_padded_nodes_neighbor_ids[b, j]
                if nid != 0 and nid in dst_set_per_batch[b]:
                    space_to_tgt_src[b, j] = 1.0
        space_to_tgt_src[src_padded_nodes_neighbor_ids == 0] = 0.0

        src_token_space = space_to_src + space_to_tgt_src

        # -------- dst 序列：对称 --------
        time_to_dst = np.zeros((batch_size, dst_len), dtype=np.float32)
        time_to_dst[:, 1:] = (cur_time - dst_padded_nodes_neighbor_times[:, 1:]) / (cur_time + 1.0 + eps)
        time_to_dst[dst_padded_nodes_neighbor_ids == 0] = 0.0

        time_to_src_dst = np.ones((batch_size, dst_len), dtype=np.float32)
        for b in range(batch_size):
            src_id = src_node_ids[b] if hasattr(src_node_ids, '__getitem__') else src_node_ids.flat[b]
            node_to_time = {}
            for k in range(src_len):
                nid = src_padded_nodes_neighbor_ids[b, k]
                if nid == 0:
                    continue
                t = (cur_time[b, 0] - src_padded_nodes_neighbor_times[b, k]) / (cur_time[b, 0] + 1.0 + eps)
                node_to_time[nid] = t
            for j in range(dst_len):
                nid = dst_padded_nodes_neighbor_ids[b, j]
                if nid == 0:
                    continue
                if nid == src_id:
                    time_to_src_dst[b, j] = 0.0
                elif nid in node_to_time:
                    time_to_src_dst[b, j] = node_to_time[nid]
        time_to_src_dst[dst_padded_nodes_neighbor_ids == 0] = 0.0

        dst_token_time = time_to_dst + time_to_src_dst

        space_to_dst = np.zeros((batch_size, dst_len), dtype=np.float32)
        space_to_dst[:, 1:] = 1.0
        space_to_dst[dst_padded_nodes_neighbor_ids == 0] = 0.0

        src_set_per_batch = [set(src_padded_nodes_neighbor_ids[b].flat) for b in range(batch_size)]
        space_to_src_dst = np.full((batch_size, dst_len), 2.0, dtype=np.float32)
        for b in range(batch_size):
            space_to_src_dst[b, dst_padded_nodes_neighbor_ids[b] == src_node_ids[b]] = 0.0
            for j in range(dst_len):
                nid = dst_padded_nodes_neighbor_ids[b, j]
                if nid != 0 and nid in src_set_per_batch[b]:
                    space_to_src_dst[b, j] = 1.0
        space_to_src_dst[dst_padded_nodes_neighbor_ids == 0] = 0.0

        dst_token_space = space_to_dst + space_to_src_dst

        return src_token_time, src_token_space, dst_token_time, dst_token_space

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        if isinstance(src_node_ids, torch.Tensor):
            src_node_ids = src_node_ids.cpu().numpy()
        if isinstance(dst_node_ids, torch.Tensor):
            dst_node_ids = dst_node_ids.cpu().numpy()
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()

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

        # 每个 token 的时间/空间标量，(B, L_src), (B, L_src), (B, L_dst), (B, L_dst)
        src_token_time, src_token_space, dst_token_time, dst_token_space = self.compute_token_time_space_scalars(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            src_padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
        )

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

        # -------- patch 级时空标量 + 重排（源/目标各自序列内，位置 0 固定）--------
        batch_size = src_token_time.shape[0]
        src_len = src_token_time.shape[1]
        dst_len = dst_token_time.shape[1]
        src_num_patches = src_len // self.patch_size
        dst_num_patches = dst_len // self.patch_size

        # patch 标量 = 该 patch 内 token 标量的均值
        src_token_time_2d = src_token_time.reshape(batch_size, src_num_patches, self.patch_size)
        src_token_space_2d = src_token_space.reshape(batch_size, src_num_patches, self.patch_size)
        patch_time_src = np.mean(src_token_time_2d, axis=2)   # (B, src_num_patches)
        patch_space_src = np.mean(src_token_space_2d, axis=2)
        key_src = patch_time_src + patch_space_src

        dst_token_time_2d = dst_token_time.reshape(batch_size, dst_num_patches, self.patch_size)
        dst_token_space_2d = dst_token_space.reshape(batch_size, dst_num_patches, self.patch_size)
        patch_time_dst = np.mean(dst_token_time_2d, axis=2)
        patch_space_dst = np.mean(dst_token_space_2d, axis=2)
        key_dst = patch_time_dst + patch_space_dst

        # 每个 batch：位置 0 不动，其余按 key 升序排列
        order_src = np.zeros((batch_size, src_num_patches), dtype=np.int64)
        for b in range(batch_size):
            order_src[b, 0] = 0
            if src_num_patches > 1:
                order_src[b, 1:] = 1 + np.argsort(key_src[b, 1:])

        order_dst = np.zeros((batch_size, dst_num_patches), dtype=np.int64)
        for b in range(batch_size):
            order_dst[b, 0] = 0
            if dst_num_patches > 1:
                order_dst[b, 1:] = 1 + np.argsort(key_dst[b, 1:])

        # 用 order 重排所有 patch 张量（在 projection 之前）
        order_src_th = torch.from_numpy(order_src).to(self.device)  # (B, src_num_patches)
        order_dst_th = torch.from_numpy(order_dst).to(self.device)
        batch_idx = torch.arange(batch_size, device=self.device)[:, None]

        src_patches_nodes_neighbor_node_raw_features = src_patches_nodes_neighbor_node_raw_features[batch_idx, order_src_th]
        src_patches_nodes_edge_raw_features = src_patches_nodes_edge_raw_features[batch_idx, order_src_th]
        src_patches_nodes_neighbor_time_features = src_patches_nodes_neighbor_time_features[batch_idx, order_src_th]
        src_patches_nodes_neighbor_co_occurrence_features = src_patches_nodes_neighbor_co_occurrence_features[batch_idx, order_src_th]

        dst_patches_nodes_neighbor_node_raw_features = dst_patches_nodes_neighbor_node_raw_features[batch_idx, order_dst_th]
        dst_patches_nodes_edge_raw_features = dst_patches_nodes_edge_raw_features[batch_idx, order_dst_th]
        dst_patches_nodes_neighbor_time_features = dst_patches_nodes_neighbor_time_features[batch_idx, order_dst_th]
        dst_patches_nodes_neighbor_co_occurrence_features = dst_patches_nodes_neighbor_co_occurrence_features[batch_idx, order_dst_th]

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

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat([src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # -------- RoPE：按重排后的序列位置生成 cos/sin 并作用于 Q、K --------
        padded_seq_len = patches_data.shape[1]
        rotary_dim = self.head_dim
        cos, sin = self.compute_rotary_embedding(padded_seq_len, rotary_dim)

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            patches_data = transformer(patches_data, cos, sin, rotary_dim)

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

    def __init__(self, attention_dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        """
        Transformer encoder with RoPE support.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param head_dim: int, dimension of each attention head (for RoPE)
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.query_projection = nn.Linear(attention_dim, attention_dim)
        self.key_projection = nn.Linear(attention_dim, attention_dim)
        self.value_projection = nn.Linear(attention_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, attention_dim)

        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int):
        """
        encode the inputs by Transformer encoder with RoPE
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :param cos: Tensor, shape (seq_len, rotary_dim//2)
        :param sin: Tensor, shape (seq_len, rotary_dim//2)
        :param rotary_dim: int, dimension of the rotary embedding
        :return:
        """
        batch_size, seq_len, embed_dim = inputs.shape

        inputs = self.norm_layers[0](inputs)

        q = self.query_projection(inputs)
        k = self.key_projection(inputs)
        v = self.value_projection(inputs)

        q = q.view(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, 2 * self.head_dim).transpose(1, 2)

        if cos.shape[0] < seq_len:
            current_len = cos.shape[0]
            needed_len = seq_len
            last_cos = cos[-1:]
            last_sin = sin[-1:]
            repeat_times = needed_len - current_len
            extended_cos = torch.cat([cos, last_cos.repeat(repeat_times, 1)], dim=0)
            extended_sin = torch.cat([sin, last_sin.repeat(repeat_times, 1)], dim=0)
            cos = extended_cos
            sin = extended_sin
        else:
            cos = cos[:seq_len]
            sin = sin[:seq_len]

        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (2 * self.head_dim) ** 0.5
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)

        attention_output = self.output_projection(context)
        outputs = inputs + self.dropout(attention_output)

        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        outputs = outputs + self.dropout(hidden_states)
        return outputs