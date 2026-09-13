import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.graph import NeighborFinder, GraphMaker
from utils.wsdmmodule import FixedSinCosEncode


# =========================
# 可调的数值尺度（必要时微调）
# =========================
TEMPORAL_INPUT_SCALE = 1.0   # 时间标量进入 TimeEncoder 前的缩放（沿用原先位置）
SPATIAL_INPUT_SCALE  = 1.0   # 空间桶/距离进入 FixedSinCosEncode 前的缩放


def recent_interact_src_to_neigh(
    memory, src_idx_l, neigh_idx_l, cur_time_l, alpha=10.0, beta=1.0
):
    """
    根据 memory 统计得到 (最近一次交互时间, 交互次数) → 组合成 temporal distance 标量。
    修复点：
      1) 对 padding (neigh==0) 直接返回 0；
      2) 去掉原本的 * eta(10000) 爆炸缩放；
      3) 结果保持在合理量级，后续可用 TEMPORAL_INPUT_SCALE 再做整体缩放。
    """
    most_recent_interact = np.zeros_like(neigh_idx_l, dtype=np.float32)
    interact_cnt = np.zeros_like(neigh_idx_l, dtype=np.float32)
    pad_mask_np = (neigh_idx_l == 0)

    for i in range(neigh_idx_l.shape[0]):
        for j in range(neigh_idx_l.shape[1]):
            if pad_mask_np[i, j]:
                # padding: 直接保留为 0
                continue
            ky = f'{src_idx_l[i]}_{neigh_idx_l[i, j]}'
            if ky in memory:
                most_recent_interact[i, j] = float(memory[ky]['recent'])
                interact_cnt[i, j] = float(memory[ky]['count'])
            else:
                most_recent_interact[i, j] = 0.0
                interact_cnt[i, j] = 0.0

    # 防止除零：分母 +1
    cur_col = cur_time_l[:, np.newaxis].astype(np.float32)
    intensity = most_recent_interact / (interact_cnt * cur_col + 1.0)        # ~[0,1]
    recent    = (cur_col - most_recent_interact) / (cur_col + 1.0)           # ~[0,1]

    final_dis = beta * intensity + alpha * recent
    # padding 位置显式清零
    final_dis[pad_mask_np] = 0.0

    # 可选全局缩放（默认 1.0）
    final_dis = final_dis * TEMPORAL_INPUT_SCALE
    return final_dis.astype(np.float32)


def get_spatial_distance(src_neigh, tgt_neigh, offset):
    """
    返回空间“桶”索引（整数），再交给编码器。
    修复点：
      1) padding (id==0) 的位置桶设为 0；
      2) 取消 *10000 的放大，避免高频折返/投影饱和；
      3) 保持输出为小整数桶，后续可再做 Embedding 或 SinCos 编码；
      4) 最后按 SPATIAL_INPUT_SCALE 做可选缩放（对 SinCos 输入的整体控制）。
    """
    mask = (src_neigh[:, :, None] == tgt_neigh[:, None, :])  # (B,L,L)
    min_indices = np.full_like(src_neigh, fill_value=10_000, dtype=np.int64)

    # 找到 src->tgt 的最小匹配下标
    for i in range(mask.shape[-1]):
        index = np.where(mask[:, :, i])
        min_indices[index[0], index[1]] = np.minimum(min_indices[index[0], index[1]], i)

    transformed = np.zeros_like(src_neigh, dtype=np.int64)

    # 根据 offset 做桶划分
    for i in range(len(offset) - 1):
        mask2 = np.logical_and(min_indices >= offset[i], min_indices < offset[i + 1])
        transformed[mask2] = i

    transformed[min_indices >= offset[-1]] = len(offset) - 1  # 最大桶

    # padding 位置 → 0 桶（专门当成“无效/空”）
    transformed[src_neigh == 0] = 0

    # 转成 float32 以便送入编码器；再做可选缩放
    transformed = transformed.astype(np.float32) * SPATIAL_INPUT_SCALE
    return transformed


class STDyGFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_finder: NeighborFinder,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu', k_hop: int = 2, num_neighbors: list = [10, 5]):
        """
        STDyGFormer model.
        """
        super(STDyGFormer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_finder = neighbor_finder
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        required_divisor = num_heads  # PyTorch MultiheadAttention 要求 embed_dim 能整除 num_heads
        lcm_unit = required_divisor // math.gcd(k_hop*0+5, required_divisor)  # num_channels 此处固定5
        if channel_embedding_dim % lcm_unit != 0:
            channel_embedding_dim = ((channel_embedding_dim + lcm_unit - 1) // lcm_unit) * lcm_unit
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        self.k_hop = k_hop

        # 规范化 num_neighbors
        if isinstance(num_neighbors, str):
            import ast
            num_neighbors = ast.literal_eval(num_neighbors)
        if isinstance(num_neighbors, int):
            num_neighbors = [num_neighbors] * k_hop
        elif isinstance(num_neighbors, (list, tuple)):
            num_neighbors = list(num_neighbors)
            if len(num_neighbors) < k_hop:
                num_neighbors.extend([num_neighbors[-1]] * (k_hop - len(num_neighbors)))
            else:
                num_neighbors = num_neighbors[:k_hop]
        else:
            raise ValueError(f"num_neighbors 必须为 int / list / tuple / str，但收到 {type(num_neighbors)}")
        self.num_neighbors = num_neighbors

        # ===== 编码器：时间改为 TimeEncoder（可学习），空间仍用 FixedSinCosEncode =====
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim)  # 替换点（原来是 FixedSinCosEncode）
        self.spatial_encoder  = FixedSinCosEncode(hidden_size=self.time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'temporal': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'spatial': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 5

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

        self.graph_maker = GraphMaker(num_neigh=self.num_neighbors, max_depth=self.k_hop)
        self.memory = {}

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, **kwargs):
        """
        compute source and destination node temporal embeddings
        """
        # ------- numpy 化 -------
        if isinstance(src_node_ids, torch.Tensor):
            src_node_ids = src_node_ids.detach().cpu().numpy().astype(np.int32)
        if isinstance(dst_node_ids, torch.Tensor):
            dst_node_ids = dst_node_ids.detach().cpu().numpy().astype(np.int32)
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy().astype(np.float32)

        # ------- k-hop 采样 -------
        src_one_hop_node, src_one_hop_t, src_node_records, src_t_records, src_eidx_g, src_offset_per_layer = self.neighbor_finder.find_k_hop(
            self.k_hop, src_node_ids, node_interact_times, self.graph_maker, self.num_neighbors)
        dst_one_hop_node, dst_one_hop_t, dst_node_records, dst_t_records, dst_eidx_g, dst_offset_per_layer = self.neighbor_finder.find_k_hop(
            self.k_hop, dst_node_ids, node_interact_times, self.graph_maker, self.num_neighbors)

        # ------- pad mask（token 级）-------
        src_token_pad = (src_node_records == 0)
        dst_token_pad = (dst_node_records == 0)

        # ------- 特征提取 -------
        # 节点/边
        src_node_features = self.node_raw_features[torch.from_numpy(src_node_records).long()]
        dst_node_features = self.node_raw_features[torch.from_numpy(dst_node_records).long()]

        src_diag_idx = np.diagonal(src_eidx_g, axis1=1, axis2=2).astype(np.int64)
        dst_diag_idx = np.diagonal(dst_eidx_g, axis1=1, axis2=2).astype(np.int64)
        idx_device = self.edge_raw_features.device
        src_edge_idx_tensor = torch.from_numpy(src_diag_idx).long().to(idx_device)
        dst_edge_idx_tensor = torch.from_numpy(dst_diag_idx).long().to(idx_device)
        src_edge_features = self.edge_raw_features[src_edge_idx_tensor]
        dst_edge_features = self.edge_raw_features[dst_edge_idx_tensor]

        # 共现
        src_neighbor_co_occurrence_features, dst_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encoder(
            src_padded_nodes_neighbor_ids=src_node_records, dst_padded_nodes_neighbor_ids=dst_node_records)

        # ===== 时间（保持统计构造，但编码器改为 TimeEncoder）=====
        src_temporal_np = recent_interact_src_to_neigh(self.memory, src_node_ids, src_node_records, node_interact_times)
        dst_temporal_np = recent_interact_src_to_neigh(self.memory, dst_node_ids, dst_node_records, node_interact_times)

        # ===== 空间 =====
        src_spatial_np  = get_spatial_distance(src_node_records, src_node_records, src_offset_per_layer)
        dst_spatial_np  = get_spatial_distance(dst_node_records, dst_node_records, dst_offset_per_layer)

        # ===== 编码：时间用 TimeEncoder，空间仍用固定 SinCos =====
        # TimeEncoder 期望 torch.Tensor，形状 (B, L)；输出 (B, L, time_feat_dim)
        src_temporal_feature = self.time_encoder(torch.from_numpy(src_temporal_np).float().to(self.device))
        dst_temporal_feature = self.time_encoder(torch.from_numpy(dst_temporal_np).float().to(self.device))

        src_spatial_feature  = self.spatial_encoder(src_spatial_np)
        dst_spatial_feature  = self.spatial_encoder(dst_spatial_np)

        # ------- 设备/类型对齐 -------
        device = self.device
        to_f32 = lambda x: x.to(device).float()
        src_node_features  = to_f32(src_node_features)
        dst_node_features  = to_f32(dst_node_features)
        src_edge_features  = to_f32(src_edge_features)
        dst_edge_features  = to_f32(dst_edge_features)
        src_temporal_feature = to_f32(src_temporal_feature)
        dst_temporal_feature = to_f32(dst_temporal_feature)
        src_spatial_feature  = to_f32(src_spatial_feature)
        dst_spatial_feature  = to_f32(dst_spatial_feature)
        src_neighbor_co_occurrence_features = to_f32(src_neighbor_co_occurrence_features)
        dst_neighbor_co_occurrence_features = to_f32(dst_neighbor_co_occurrence_features)

        # ------- 显式置零：所有通道在 padding 位置都置零 -------
        src_pad_mask = torch.from_numpy(src_token_pad).to(device).bool()  # (B,L)
        dst_pad_mask = torch.from_numpy(dst_token_pad).to(device).bool()  # (B,L)

        src_node_features[src_pad_mask]  = 0
        src_edge_features[src_pad_mask]  = 0
        src_temporal_feature[src_pad_mask] = 0
        src_spatial_feature[src_pad_mask]  = 0
        src_neighbor_co_occurrence_features[src_pad_mask] = 0

        dst_node_features[dst_pad_mask]  = 0
        dst_edge_features[dst_pad_mask]  = 0
        dst_temporal_feature[dst_pad_mask] = 0
        dst_spatial_feature[dst_pad_mask]  = 0
        dst_neighbor_co_occurrence_features[dst_pad_mask] = 0

        # ------- patch 化（5 通道）-------
        src_patches = self.get_patches_with_spatial(src_node_features, src_edge_features, src_temporal_feature, src_spatial_feature, src_neighbor_co_occurrence_features)
        dst_patches = self.get_patches_with_spatial(dst_node_features, dst_edge_features, dst_temporal_feature, dst_spatial_feature, dst_neighbor_co_occurrence_features)

        # ------- patch 级 padding mask -------
        def build_patch_mask(token_pad, patch_size):
            B, L = token_pad.shape
            usable_len = (L // patch_size) * patch_size
            token_pad = token_pad[:, :usable_len]
            num_patches = usable_len // patch_size
            token_pad = torch.from_numpy(token_pad.reshape(B, num_patches, patch_size)).to(self.device)
            patch_pad = torch.all(token_pad, dim=-1)  # (B, num_patches)
            return patch_pad

        src_patch_pad = build_patch_mask(src_token_pad, self.patch_size)
        dst_patch_pad = build_patch_mask(dst_token_pad, self.patch_size)

        # ------- 线性投影 -------
        src_proj = [self.projection_layer['node'](src_patches[0]),
                    self.projection_layer['edge'](src_patches[1]),
                    self.projection_layer['temporal'](src_patches[2]),
                    self.projection_layer['spatial'](src_patches[3]),
                    self.projection_layer['neighbor_co_occurrence'](src_patches[4])]
        dst_proj = [self.projection_layer['node'](dst_patches[0]),
                    self.projection_layer['edge'](dst_patches[1]),
                    self.projection_layer['temporal'](dst_patches[2]),
                    self.projection_layer['spatial'](dst_patches[3]),
                    self.projection_layer['neighbor_co_occurrence'](dst_patches[4])]

        # ------- 拼接（通道）-------
        src_cat = torch.cat(src_proj, dim=-1)  # (B, P, C*dim)
        dst_cat = torch.cat(dst_proj, dim=-1)

        # ------- Transformer（带 mask）-------
        for transformer in self.transformers:
            src_cat = transformer(src_cat, key_padding_mask=src_patch_pad)  # 屏蔽全 pad 的 patch
            dst_cat = transformer(dst_cat, key_padding_mask=dst_patch_pad)

        # ------- 聚合 & 输出 -------
        src_emb = torch.mean(src_cat, dim=1)
        dst_emb = torch.mean(dst_cat, dim=1)
        src_node_embeddings = self.output_layer(src_emb)
        dst_node_embeddings = self.output_layer(dst_emb)
        return src_node_embeddings, dst_node_embeddings

    def get_patches(self, node_features, edge_features, temporal_features, spatial_features, neighbor_co_occurrence_features):
        batch_size, seq_len, feat_dim = node_features.shape
        seq_len = int(seq_len)
        patch_size = self.patch_size
        if isinstance(patch_size, torch.Tensor):
            patch_size = int(patch_size.item())
        patch_size = int(patch_size)
        if seq_len % patch_size != 0:
            usable_len = (seq_len // patch_size) * patch_size
            node_features = node_features[:, :usable_len, :]
            edge_features = edge_features[:, :usable_len, :]
            temporal_features = temporal_features[:, :usable_len, :]
            spatial_features = spatial_features[:, :usable_len, :]
            neighbor_co_occurrence_features = neighbor_co_occurrence_features[:, :usable_len, :]
            seq_len = usable_len
        num_patches = seq_len // patch_size
        def patchify(x):
            return x.reshape(batch_size, num_patches, -1)
        return [patchify(node_features), patchify(edge_features), patchify(temporal_features), patchify(spatial_features), patchify(neighbor_co_occurrence_features)]

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
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

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device)
        )
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0
        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches_with_spatial(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                                 temporal_features: torch.Tensor, spatial_features: torch.Tensor,
                                 neighbor_co_occurrence_features: torch.Tensor):
        batch_size, seq_len, _ = node_features.shape
        patch_size = int(self.patch_size) if not isinstance(self.patch_size, torch.Tensor) else int(self.patch_size.item())

        if seq_len % patch_size != 0:
            usable_len = (seq_len // patch_size) * patch_size
            node_features = node_features[:, :usable_len, :]
            edge_features = edge_features[:, :usable_len, :]
            temporal_features = temporal_features[:, :usable_len, :]
            spatial_features = spatial_features[:, :usable_len, :]
            neighbor_co_occurrence_features = neighbor_co_occurrence_features[:, :usable_len, :]
            seq_len = usable_len

        num_patches = seq_len // patch_size

        def patchify(x):
            return x.reshape(batch_size, num_patches, -1)

        return [
            patchify(node_features),
            patchify(edge_features),
            patchify(temporal_features),
            patchify(spatial_features),
            patchify(neighbor_co_occurrence_features)
        ]

    def set_neighbor_sampler(self, neighbor_sampler):
        from utils.graph import NeighborFinder
        if isinstance(neighbor_sampler, NeighborFinder):
            self.neighbor_finder = neighbor_sampler
            return
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self, 'neighbor_sampler') and \
           hasattr(self.neighbor_sampler, 'sample_neighbor_strategy') and \
           self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim)
        )

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)
            ).float().to(self.device)
            src_padded_nodes_appearances.append(torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)
            ).float().to(self.device)
            dst_padded_nodes_appearances.append(torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
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

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        """
        super(TransformerEncoder, self).__init__()
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

    def forward(self, inputs: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        encode the inputs by Transformer encoder
        :param inputs: (B, P, D)
        :param key_padding_mask: (B, P)  True=pad，会在注意力中屏蔽
        """
        # PyTorch MHA 期望 (P, B, D)
        transposed_inputs = inputs.transpose(0, 1)
        transposed_inputs = self.norm_layers[0](transposed_inputs)

        # 注意：key_padding_mask 的 shape 应为 (B, P)
        attn_out, _ = self.multi_head_attention(
            query=transposed_inputs, key=transposed_inputs, value=transposed_inputs,
            key_padding_mask=key_padding_mask
        )

        hidden_states = attn_out.transpose(0, 1)
        outputs = inputs + self.dropout(hidden_states)

        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        outputs = outputs + self.dropout(hidden_states)
        return outputs
