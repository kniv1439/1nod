import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class DyGFormerCC(nn.Module):

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        channel_embedding_dim: int,
        patch_size: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_input_sequence_length: int = 512,
        device: str = "cpu",
        dp_rank: int = 16,
        alpha: float = 0.5,
        merge_size: int = 1,
    ):
        """
        DyGFormer with modified CARD-style attention.

        Transformer input:
            patches_data: [B, T, 4, D]

        where:
            B = batch size
            T = src_num_patches + dst_num_patches
            4 = node / edge / time / neighbor_co_occurrence channels
            D = channel_embedding_dim
        """
        super(DyGFormerCC, self).__init__()

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
            device=self.device,
        )

        self.projection_layer = nn.ModuleDict({
            "node": nn.Linear(
                in_features=self.patch_size * self.node_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
            "edge": nn.Linear(
                in_features=self.patch_size * self.edge_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
            "time": nn.Linear(
                in_features=self.patch_size * self.time_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
            "neighbor_co_occurrence": nn.Linear(
                in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
        })

        self.num_channels = 4
        max_token_num = 2 * (self.max_input_sequence_length // self.patch_size) + 1

        self.transformers = nn.ModuleList([
            TransformerEncoder(
                channel_embedding_dim=self.channel_embedding_dim,
                num_heads=self.num_heads,
                num_channels=self.num_channels,
                dropout=self.dropout,
                dp_rank=dp_rank,
                alpha=alpha,
                merge_size=merge_size,
                max_token_num=max_token_num,
            )
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=self.node_feat_dim,
            bias=True,
        )

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
    ):
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
            )

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
            )

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                nodes_edge_ids_list=src_nodes_edge_ids_list,
                nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                patch_size=self.patch_size,
                max_input_sequence_length=self.max_input_sequence_length,
            )

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                nodes_edge_ids_list=dst_nodes_edge_ids_list,
                nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                patch_size=self.patch_size,
                max_input_sequence_length=self.max_input_sequence_length,
            )

        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            )

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )

        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
            )

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
            )

        src_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
            src_patches_nodes_neighbor_node_raw_features
        )
        src_patches_nodes_edge_raw_features = self.projection_layer["edge"](
            src_patches_nodes_edge_raw_features
        )
        src_patches_nodes_neighbor_time_features = self.projection_layer["time"](
            src_patches_nodes_neighbor_time_features
        )
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer["neighbor_co_occurrence"](
            src_patches_nodes_neighbor_co_occurrence_features
        )

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
            dst_patches_nodes_neighbor_node_raw_features
        )
        dst_patches_nodes_edge_raw_features = self.projection_layer["edge"](
            dst_patches_nodes_edge_raw_features
        )
        dst_patches_nodes_neighbor_time_features = self.projection_layer["time"](
            dst_patches_nodes_neighbor_time_features
        )
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer["neighbor_co_occurrence"](
            dst_patches_nodes_neighbor_co_occurrence_features
        )

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        patches_nodes_neighbor_node_raw_features = torch.cat(
            [src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features],
            dim=1,
        )
        patches_nodes_edge_raw_features = torch.cat(
            [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features],
            dim=1,
        )
        patches_nodes_neighbor_time_features = torch.cat(
            [src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features],
            dim=1,
        )
        patches_nodes_neighbor_co_occurrence_features = torch.cat(
            [src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features],
            dim=1,
        )

        patches_data = [
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
        ]

        # [B, T, 4, D]
        patches_data = torch.stack(patches_data, dim=2)

        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        src_patches_data = patches_data[:, :src_num_patches, :, :]
        dst_patches_data = patches_data[:, src_num_patches:src_num_patches + dst_num_patches, :, :]

        src_patches_data = torch.mean(src_patches_data, dim=1).reshape(batch_size, -1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1).reshape(batch_size, -1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids_list: list,
        nodes_edge_ids_list: list,
        nodes_neighbor_times_list: list,
        patch_size: int = 1,
        max_input_sequence_length: int = 256,
    ):
        assert max_input_sequence_length - 1 > 0, \
            "Maximal number of neighbors for each node should be greater than 1!"

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
                padded_nodes_neighbor_ids[idx, 1:len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1:len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1:len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(
        self,
        node_interact_times: np.ndarray,
        padded_nodes_neighbor_ids: np.ndarray,
        padded_nodes_edge_ids: np.ndarray,
        padded_nodes_neighbor_times: np.ndarray,
        time_encoder: TimeEncoder,
    ):
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()

        padded_nodes_neighbor_node_raw_features = self.node_raw_features[
            torch.from_numpy(padded_nodes_neighbor_ids)
        ]

        padded_nodes_edge_raw_features = self.edge_raw_features[
            torch.from_numpy(padded_nodes_edge_ids)
        ]

        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(
                node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
            ).float().to(self.device)
        )

        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return (
            padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features,
        )

    def get_patches(
        self,
        padded_nodes_neighbor_node_raw_features: torch.Tensor,
        padded_nodes_edge_raw_features: torch.Tensor,
        padded_nodes_neighbor_time_features: torch.Tensor,
        padded_nodes_neighbor_co_occurrence_features: torch.Tensor,
        patch_size: int = 1,
    ):
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0

        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        patches_nodes_neighbor_node_raw_features = []
        patches_nodes_edge_raw_features = []
        patches_nodes_neighbor_time_features = []
        patches_nodes_neighbor_co_occurrence_features = []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size

            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_edge_raw_features.append(
                padded_nodes_edge_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_time_features.append(
                padded_nodes_neighbor_time_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_co_occurrence_features.append(
                padded_nodes_neighbor_co_occurrence_features[:, start_idx:end_idx, :]
            )

        batch_size = len(padded_nodes_neighbor_node_raw_features)

        patches_nodes_neighbor_node_raw_features = torch.stack(
            patches_nodes_neighbor_node_raw_features,
            dim=1,
        ).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)

        patches_nodes_edge_raw_features = torch.stack(
            patches_nodes_edge_raw_features,
            dim=1,
        ).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)

        patches_nodes_neighbor_time_features = torch.stack(
            patches_nodes_neighbor_time_features,
            dim=1,
        ).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(
            patches_nodes_neighbor_co_occurrence_features,
            dim=1,
        ).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return (
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
        )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = "cpu"):
        super(NeighborCooccurrenceEncoder, self).__init__()

        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

    def count_nodes_appearances(
        self,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
    ):
        src_padded_nodes_appearances = []
        dst_padded_nodes_appearances = []

        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(
            src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids,
        ):
            src_unique_keys, src_inverse_indices, src_counts = np.unique(
                src_padded_node_neighbor_ids,
                return_inverse=True,
                return_counts=True,
            )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(
                src_counts[src_inverse_indices]
            ).float().to(self.device)

            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                dst_padded_node_neighbor_ids,
                return_inverse=True,
                return_counts=True,
            )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                dst_counts[dst_inverse_indices]
            ).float().to(self.device)

            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                src_padded_node_neighbor_ids.copy()
            ).apply_(
                lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)
            ).float().to(self.device)

            src_padded_nodes_appearances.append(
                torch.stack(
                    [
                        src_padded_node_neighbor_counts_in_src,
                        src_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )

            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(
                dst_padded_node_neighbor_ids.copy()
            ).apply_(
                lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)
            ).float().to(self.device)

            dst_padded_nodes_appearances.append(
                torch.stack(
                    [
                        dst_padded_node_neighbor_counts_in_src,
                        dst_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(
        self,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
    ):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
        )

        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)
        ).sum(dim=2)

        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)
        ).sum(dim=2)

        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class CARDChannelSeqDPAttention(nn.Module):
    """
    Modified CARD Channel block.

    Input:
        src: [B, T, C, D]

    Design:
        1. Channel attention is performed over C channels.
        2. Dynamic projection is applied over sequence dimension T.
        3. Hidden-dimension attention is unchanged.
        4. BatchNorm is replaced by LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_channels: int,
        max_token_num: int,
        dp_rank: int = 16,
        alpha: float = 0.5,
        merge_size: int = 1,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_heads % merge_size == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_channels = num_channels
        self.max_token_num = max_token_num
        self.dp_rank = dp_rank
        self.merge_size = merge_size

        d_ff = d_ff or 4 * d_model

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.dp_k = nn.Linear(self.head_dim, dp_rank)
        self.dp_v = nn.Linear(self.head_dim, dp_rank)

        self.norm_post1 = nn.LayerNorm(d_model)
        self.norm_post2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)

        self.ff_1 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )

        self.ff_2 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )

        ema_size = max(num_channels, max_token_num, dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size))
        ema_matrix[0][0] = 1.0

        for i in range(1, ema_size):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i - 1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha

        self.register_buffer("ema_matrix", ema_matrix)

    def ema(self, src: torch.Tensor) -> torch.Tensor:
        length = src.shape[-2]
        ema_matrix = self.ema_matrix[:length, :length]
        return torch.einsum("...ad,ga->...gd", src, ema_matrix)

    def sequence_dynamic_projection(self, src: torch.Tensor, mlp: nn.Linear) -> torch.Tensor:
        """
        Dynamic projection over sequence dimension T.

        Input:
            src: [B, T, heads, C, head_dim]

        Output:
            src_dp: [B, dp_rank, heads, C, head_dim]
        """
        weights = mlp(src)              # [B, T, heads, C, dp_rank]
        weights = F.softmax(weights, dim=-1)

        src_dp = torch.einsum(
            "bthcd,bthcr->brhcd",
            src,
            weights,
        )

        return src_dp

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, nvars, heads, H, head_dim]

        Output:
            x: [B * nvars, H, d_model]
        """
        B, nvars, heads, H, head_dim = x.shape

        x = rearrange(
            x.reshape(B * nvars, heads * H, head_dim),
            "bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d",
            hl1=heads // self.merge_size,
            hl2=H,
            hl3=self.merge_size,
        )

        x = x.reshape(B * nvars, H, heads * head_dim)
        return x

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: [B, T, C, D]
        """
        B, T, C, D = src.shape

        assert C == self.num_channels, \
            f"Expected num_channels={self.num_channels}, but got C={C}."

        qkv = self.qkv(src)
        qkv = qkv.reshape(B, T, C, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)

        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, T, heads, C, head_dim]

        k_dp = self.sequence_dynamic_projection(k, self.dp_k)
        v_dp = self.sequence_dynamic_projection(v, self.dp_v)

        q_ema = self.ema(q)          # EMA over C
        k_dp_ema = self.ema(k_dp)    # EMA over C

        attn_score_channel = torch.einsum(
            "bthcd,brhfd->btrhcf",
            q_ema,
            k_dp_ema,
        ) / (self.head_dim ** 0.5)

        attn_channel = self.attn_dropout(F.softmax(attn_score_channel, dim=-1))

        output_along_channel = torch.einsum(
            "btrhcf,brhfd->btrhcd",
            attn_channel,
            v_dp,
        )

        output_along_channel = output_along_channel.mean(dim=2)

        attn_score_hidden = torch.einsum(
            "bthce,bthcf->bthef",
            q,
            k,
        ) / (C ** 0.5)

        attn_hidden = self.attn_dropout(F.softmax(attn_score_hidden, dim=-1))

        output_along_hidden = torch.einsum(
            "bthef,bthcf->bthce",
            attn_hidden,
            v,
        )

        output1 = self.merge_heads(output_along_channel)
        output2 = self.merge_heads(output_along_hidden)

        output1 = self.norm_post1(output1).reshape(B, T, C, self.d_model)
        output2 = self.norm_post2(output2).reshape(B, T, C, self.d_model)

        src2 = self.ff_1(output1) + self.ff_2(output2)

        out = src + src2
        out = self.norm_attn(out)

        return out


class CARDTokenAttention(nn.Module):
    """
    CARD Token block.

    Input:
        src: [B, C, T, D]

    Design:
        1. Full EMA attention over patch/token dimension T.
        2. Hidden-dimension attention over head_dim.
        3. BatchNorm is replaced by LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_channels: int,
        max_token_num: int,
        alpha: float = 0.5,
        merge_size: int = 1,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_heads % merge_size == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_channels = num_channels
        self.max_token_num = max_token_num
        self.merge_size = merge_size

        d_ff = d_ff or 4 * d_model

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_post1 = nn.LayerNorm(d_model)
        self.norm_post2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)

        self.ff_1 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )

        self.ff_2 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )

        ema_size = max(num_channels, max_token_num)
        ema_matrix = torch.zeros((ema_size, ema_size))
        ema_matrix[0][0] = 1.0

        for i in range(1, ema_size):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i - 1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha

        self.register_buffer("ema_matrix", ema_matrix)

    def ema(self, src: torch.Tensor) -> torch.Tensor:
        length = src.shape[-2]
        ema_matrix = self.ema_matrix[:length, :length]
        return torch.einsum("...ad,ga->...gd", src, ema_matrix)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, nvars, heads, H, head_dim = x.shape

        x = rearrange(
            x.reshape(B * nvars, heads * H, head_dim),
            "bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d",
            hl1=heads // self.merge_size,
            hl2=H,
            hl3=self.merge_size,
        )

        x = x.reshape(B * nvars, H, heads * head_dim)
        return x

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: [B, C, T, D]
        """
        B, C, T, D = src.shape

        assert C == self.num_channels, \
            f"Expected num_channels={self.num_channels}, but got C={C}."

        qkv = self.qkv(src)
        qkv = qkv.reshape(B, C, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)

        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, C, heads, T, head_dim]

        attn_score_token = torch.einsum(
            "bched,bchfd->bchef",
            self.ema(q),
            self.ema(k),
        ) / (self.head_dim ** 0.5)

        attn_token = self.attn_dropout(F.softmax(attn_score_token, dim=-1))

        output_along_token = torch.einsum(
            "bchef,bchfd->bched",
            attn_token,
            v,
        )

        attn_score_hidden = torch.einsum(
            "bchae,bchaf->bchef",
            q,
            k,
        ) / (T ** 0.5)

        attn_hidden = self.attn_dropout(F.softmax(attn_score_hidden, dim=-1))

        output_along_hidden = torch.einsum(
            "bchef,bchaf->bchae",
            attn_hidden,
            v,
        )

        output1 = self.merge_heads(output_along_token)
        output2 = self.merge_heads(output_along_hidden)

        output1 = self.norm_post1(output1).reshape(B, C, T, self.d_model)
        output2 = self.norm_post2(output2).reshape(B, C, T, self.d_model)

        src2 = self.ff_1(output1) + self.ff_2(output2)

        out = src + src2
        out = self.norm_attn(out)

        return out


class TransformerEncoder(nn.Module):
    """
    DyGFormer-style Pre-LN Transformer encoder
    with modified CARD-style attention module.

    Input:
        inputs: [B, T, C, D]

    Outer Transformer structure:
        Z = X + Dropout(Attention(LN1(X)))
        F = W2(Dropout(GELU(W1(LN2(Z)))))
        Y = Z + Dropout(F)

    Inner attention module:
        Channel block:
            - channel attention over C=4
            - dynamic projection over sequence dimension T
            - hidden-dimension attention over head_dim

        Token block:
            - full EMA attention over token dimension T
            - hidden-dimension attention over head_dim
    """

    def __init__(
        self,
        channel_embedding_dim: int,
        num_heads: int,
        num_channels: int,
        dropout: float = 0.1,
        dp_rank: int = 16,
        alpha: float = 0.5,
        merge_size: int = 1,
        max_token_num: int = 128,
    ):
        super(TransformerEncoder, self).__init__()

        d_ff = 4 * channel_embedding_dim

        self.attn_channel = CARDChannelSeqDPAttention(
            d_model=channel_embedding_dim,
            n_heads=num_heads,
            num_channels=num_channels,
            max_token_num=max_token_num,
            dp_rank=dp_rank,
            alpha=alpha,
            merge_size=merge_size,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.attn_token = CARDTokenAttention(
            d_model=channel_embedding_dim,
            n_heads=num_heads,
            num_channels=num_channels,
            max_token_num=max_token_num,
            alpha=alpha,
            merge_size=merge_size,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.attn_mlp = nn.Linear(channel_embedding_dim, channel_embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(channel_embedding_dim),
            nn.LayerNorm(channel_embedding_dim),
        ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(channel_embedding_dim, d_ff),
            nn.Linear(d_ff, channel_embedding_dim),
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, T, C, D]
        """

        # 1. Pre-LN before CARD-style attention.
        attn_inputs = self.norm_layers[0](inputs)

        # 2. Channel block.
        output_channel = self.attn_channel(attn_inputs)

        # 3. Token block.
        output_token = self.attn_token(
            output_channel.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        # 4. Attention fusion.
        attn_output = self.attn_mlp(output_channel + output_token)

        # 5. DyGFormer-style attention residual.
        z = inputs + self.dropout(attn_output)

        # 6. Pre-LN FFN.
        f = self.linear_layers[1](
            self.dropout(
                F.gelu(
                    self.linear_layers[0](
                        self.norm_layers[1](z)
                    )
                )
            )
        )

        # 7. FFN residual.
        outputs = z + self.dropout(f)

        return outputs