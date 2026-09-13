import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from models.LDyGFormer import (
    NeighborCooccurrenceEncoder,
    TransformerEncoder,
)
from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.Other import FourierLayer, series_decomp_multi
from utils.rotary import apply_rotary_emb
from utils.RoPE import LlamaDynamicNTKScalingRotaryEmbedding


class RLDyGFormer(nn.Module):
    """RoPE-enhanced LDyGFormer.

    在 Fourier 处理后的 patch 序列上应用动态 NTK 缩放的 RoPE，随后按原 LDyGFormer 流程进入 TransformerEncoder 叠层。其余流程不变。"""

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        channel_embedding_dim: int,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        device: str = "cpu",
    ):
        super(RLDyGFormer, self).__init__()

        # ---------- 与 LDyGFormer 相同的成员 ----------
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

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim).to(device)

        # 邻居共现编码器
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device,
        ).to(device)

        # 通道投影
        self.projection_layer = nn.ModuleDict(
            {
                "node": nn.Linear(
                    in_features=self.patch_size * self.node_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ).to(device),
                "edge": nn.Linear(
                    in_features=self.patch_size * self.edge_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ).to(device),
                "time": nn.Linear(
                    in_features=self.patch_size * self.time_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ).to(device),
                "neighbor_co_occurrence": nn.Linear(
                    in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ).to(device),
            }
        )

        # Fourier 处理层
        self.fourier_layer = FourierLayer(
            pred_len=self.max_input_sequence_length,
            k=4,
            low_freq=1,
            output_attention=False,
        )
        self.decomp = series_decomp_multi(kernel_size=[25])

        # 重要维度
        self.num_channels = 4
        self.attention_dim = self.num_channels * self.channel_embedding_dim
        assert (
            (self.attention_dim // self.num_heads) % 2 == 0
        ), "attention_dim / num_heads 必须是偶数 (2*head_dim)"
        self.head_dim = self.attention_dim // self.num_heads // 2  # per-head half dim

        # RoPE 缓存
        self._rotary_emb_cache = {}

        # TransformerEncoder 堆栈（复用原 LDyGFormer 实现）
        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(
                    attention_dim=self.attention_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                ).to(device)
                for _ in range(self.num_layers)
            ]
        )

        # 输出层
        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=self.node_feat_dim,
            bias=True,
        ).to(device)

        self.to(device)

    # ---------------- Fourier & RoPE utils ----------------
    def fourier_transform_processing(self, features: torch.Tensor) -> torch.Tensor:
        """趋势 + 傅里叶季节性 + 残差加和，保持长度一致。"""
        _, trend = self.decomp(features)
        seasonality, _ = self.fourier_layer(features)
        seasonality = seasonality[:, : features.shape[1], :]
        return features + seasonality + trend

    def _get_rotary_emb(self, seq_len: int, rotary_dim: int):
        if rotary_dim not in self._rotary_emb_cache:
            self._rotary_emb_cache[rotary_dim] = LlamaDynamicNTKScalingRotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=max(seq_len, self.max_input_sequence_length),
                base=10000,
                device=self.device,
                scaling_factor=1.0,
            ).to(self.device)
        return self._rotary_emb_cache[rotary_dim]

    def compute_rotary_embedding(self, seq_len: int, rotary_dim: int):
        rotary_emb = self._get_rotary_emb(seq_len, rotary_dim)
        dummy = torch.zeros(1, 1, seq_len, rotary_dim, device=self.device, dtype=torch.float32)
        cos_cached, sin_cached = rotary_emb(dummy, seq_len=seq_len)
        cos_full = cos_cached.squeeze(0).squeeze(0)
        sin_full = sin_cached.squeeze(0).squeeze(0)
        cos = cos_full[:, : rotary_dim // 2].contiguous()
        sin = sin_full[:, : rotary_dim // 2].contiguous()
        return cos, sin

    # ---------------- 主计算 ----------------
    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        **kwargs,
    ):
        """流程基本同 LDyGFormer，只在 Transformer 前加入 RoPE"""

        # -------- 获取一跳邻居 --------
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=src_node_ids, node_interact_times=node_interact_times
        )
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=dst_node_ids, node_interact_times=node_interact_times
        )

        # -------- 补齐序列 --------
        src_pad_ids, src_pad_eids, src_pad_times = self.pad_sequences(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
            nodes_edge_ids_list=src_nodes_edge_ids_list,
            nodes_neighbor_times_list=src_nodes_neighbor_times_list,
            patch_size=self.patch_size,
            max_input_sequence_length=self.max_input_sequence_length,
        )
        dst_pad_ids, dst_pad_eids, dst_pad_times = self.pad_sequences(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
            nodes_edge_ids_list=dst_nodes_edge_ids_list,
            nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
            patch_size=self.patch_size,
            max_input_sequence_length=self.max_input_sequence_length,
        )

        # -------- 邻居共现特征 --------
        src_co_occ, dst_co_occ = self.neighbor_co_occurrence_encoder(
            src_padded_nodes_neighbor_ids=src_pad_ids,
            dst_padded_nodes_neighbor_ids=dst_pad_ids,
        )

        # -------- 基础特征 --------
        src_node_raw, src_edge_raw, src_time_feat = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=src_pad_ids,
            padded_nodes_edge_ids=src_pad_eids,
            padded_nodes_neighbor_times=src_pad_times,
            time_encoder=self.time_encoder,
        )
        dst_node_raw, dst_edge_raw, dst_time_feat = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=dst_pad_ids,
            padded_nodes_edge_ids=dst_pad_eids,
            padded_nodes_neighbor_times=dst_pad_times,
            time_encoder=self.time_encoder,
        )

        # -------- Patch 切分 --------
        (
            src_patch_node,
            src_patch_edge,
            src_patch_time,
            src_patch_coocc,
        ) = self.get_patches(
            padded_nodes_neighbor_node_raw_features=src_node_raw,
            padded_nodes_edge_raw_features=src_edge_raw,
            padded_nodes_neighbor_time_features=src_time_feat,
            padded_nodes_neighbor_co_occurrence_features=src_co_occ,
            patch_size=self.patch_size,
        )
        (
            dst_patch_node,
            dst_patch_edge,
            dst_patch_time,
            dst_patch_coocc,
        ) = self.get_patches(
            padded_nodes_neighbor_node_raw_features=dst_node_raw,
            padded_nodes_edge_raw_features=dst_edge_raw,
            padded_nodes_neighbor_time_features=dst_time_feat,
            padded_nodes_neighbor_co_occurrence_features=dst_co_occ,
            patch_size=self.patch_size,
        )

        # -------- 通道投影并拼接 --------
        src_patch_node = self.projection_layer["node"](src_patch_node)
        src_patch_edge = self.projection_layer["edge"](src_patch_edge)
        src_patch_time = self.projection_layer["time"](src_patch_time)
        src_patch_coocc = self.projection_layer["neighbor_co_occurrence"](src_patch_coocc)

        dst_patch_node = self.projection_layer["node"](dst_patch_node)
        dst_patch_edge = self.projection_layer["edge"](dst_patch_edge)
        dst_patch_time = self.projection_layer["time"](dst_patch_time)
        dst_patch_coocc = self.projection_layer["neighbor_co_occurrence"](dst_patch_coocc)

        batch_size = len(src_patch_node)
        src_num_patches = src_patch_node.shape[1]
        dst_num_patches = dst_patch_node.shape[1]

        patches_data = torch.cat(
            [
                torch.cat(
                    [src_patch_node, dst_patch_node], dim=1
                ),  # node
                torch.cat([src_patch_edge, dst_patch_edge], dim=1),  # edge
                torch.cat([src_patch_time, dst_patch_time], dim=1),  # time
                torch.cat([src_patch_coocc, dst_patch_coocc], dim=1),  # co-occ
            ],
            dim=2,
        )  # (batch, total_patches, num_channels * chan_emb_dim)

        # 上一步其实把四个通道直接 concat 在 dim=2，不对；应当 stack then reshape，如 LDyGFormer
        # 重新生成 patches_data 与原实现一致
        patches_list = [
            torch.cat([src_patch_node, dst_patch_node], dim=1),
            torch.cat([src_patch_edge, dst_patch_edge], dim=1),
            torch.cat([src_patch_time, dst_patch_time], dim=1),
            torch.cat([src_patch_coocc, dst_patch_coocc], dim=1),
        ]
        patches_data = torch.stack(patches_list, dim=2).reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.num_channels * self.channel_embedding_dim,
        )

        # -------- Fourier --------
        patches_data = self.fourier_transform_processing(patches_data)

        # -------- RoPE --------
        seq_len = patches_data.shape[1]
        rotary_dim = self.head_dim  # 与 RLDIFFDyG 保持一致
        cos, sin = self.compute_rotary_embedding(seq_len, rotary_dim)

        patches_4d = patches_data.view(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        patches_4d = apply_rotary_emb(patches_4d, cos, sin)
        patches_data = patches_4d.reshape(batch_size, seq_len, self.attention_dim)

        # -------- Transformer Encoder 堆栈 --------
        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        # -------- 池化聚合 & 输出 --------
        src_patches_data = patches_data[:, : src_num_patches, :].mean(dim=1)
        dst_patches_data = patches_data[:, src_num_patches :, :].mean(dim=1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)
        return src_node_embeddings, dst_node_embeddings

    # 其余辅助函数直接复用 LDyGFormer 实现（通过 from ... import ...）
    pad_sequences = staticmethod(
        getattr(__import__("models.LDyGFormer", fromlist=["LDyGFormer"]).LDyGFormer, "pad_sequences")
    )
    get_features = staticmethod(
        getattr(__import__("models.LDyGFormer", fromlist=["LDyGFormer"]).LDyGFormer, "get_features")
    )
    get_patches = staticmethod(
        getattr(__import__("models.LDyGFormer", fromlist=["LDyGFormer"]).LDyGFormer, "get_patches")
    )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def forward(self, batch):
        src_node_ids = batch["src_node_ids"]
        dst_node_ids = batch["dst_node_ids"]
        node_interact_times = batch["node_interact_times"]
        return self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times
        )
