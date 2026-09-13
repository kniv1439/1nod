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
        momentum: float = 0.1,
    ):
        """
        DyGFormer feature pipeline with pure CARD attention replacing only the
        original DyGFormer MultiheadAttention operator.

        Before each Transformer layer, the four feature channels are flattened
        exactly as in original DyGFormer:
            patches_data: [B, T, 4 * D]

        CARDPureAttention temporarily restores [B, T, 4, D] internally for
        channel/token attention and then returns [B, T, 4 * D].

        The outer Transformer architecture remains DyGFormer-style Pre-LN:
            LN -> CARDPureAttention -> residual
            LN -> original FFN -> residual
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
        self.attention_dim = self.num_channels * self.channel_embedding_dim
        max_token_num = 2 * (self.max_input_sequence_length // self.patch_size) + 1

        # Keep the original DyGFormer Transformer architecture.  Only the
        # MultiheadAttention operator is replaced by pure CARD attention.
        self.transformers = nn.ModuleList([
            TransformerEncoder(
                attention_dim=self.attention_dim,
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

        # First keep the four DyGFormer feature channels explicit:
        # [B, T, 4, D].  Then flatten them exactly as in the original
        # DyGFormer Transformer input: [B, T, 4D].
        patches_data = torch.stack(patches_data, dim=2)
        patches_data = patches_data.reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.attention_dim,
        )

        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        src_patches_data = patches_data[:, :src_num_patches, :]
        dst_patches_data = patches_data[:, src_num_patches:src_num_patches + dst_num_patches, :]

        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

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



class CARDPureAxisAttention(nn.Module):
    """
    Pure CARD attention core for a single axis.

    This module intentionally contains NO FFN, NO LayerNorm/BatchNorm and
    NO residual connection.  It only performs:
        QKV projection -> EMA -> CARD axis attention -> hidden-dimension
        attention -> head merge -> branch summation.

    Input/output:
        src: [B, N, H, D]

    Modes:
        over_hidden=False:
            H is the token axis T and full T x T attention is used.

        over_hidden=True:
            H is the channel axis C and dynamic projection C -> r is used,
            giving C x r attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_channels: int,
        max_token_num: int,
        over_hidden: bool = False,
        dp_rank: int = 16,
        alpha: float = 0.5,
        merge_size: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        if n_heads % merge_size != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by merge_size ({merge_size})."
            )
        if dp_rank <= 0:
            raise ValueError(f"dp_rank must be positive, but got {dp_rank}.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_channels = num_channels
        self.max_token_num = max_token_num
        self.over_hidden = over_hidden
        self.dp_rank = dp_rank
        self.merge_size = merge_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # Used only by the channel-attention branch (C -> r).
        self.dp_k = nn.Linear(self.head_dim, dp_rank)
        self.dp_v = nn.Linear(self.head_dim, dp_rank)

        # Fixed causal EMA matrix.  It is sliced to the active axis length.
        ema_size = max(num_channels, max_token_num, dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size), dtype=torch.float32)
        ema_matrix[0, 0] = 1.0

        for i in range(1, ema_size):
            ema_matrix[i, :i] = ema_matrix[i - 1, :i] * (1.0 - alpha)
            ema_matrix[i, i] = alpha

        self.register_buffer("ema_matrix", ema_matrix)

    def ema(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: [B, N, heads, A, head_dim]
        """
        axis_length = src.shape[-2]
        if axis_length > self.ema_matrix.shape[0]:
            raise ValueError(
                f"EMA axis length {axis_length} exceeds precomputed size "
                f"{self.ema_matrix.shape[0]}."
            )

        ema_matrix = self.ema_matrix[:axis_length, :axis_length]
        return torch.einsum("bnhad,ga->bnhgd", src, ema_matrix)

    def dynamic_projection(
        self,
        src: torch.Tensor,
        mlp: nn.Linear,
    ) -> torch.Tensor:
        """
        Dynamic projection over the active H axis.

        src:     [B, N, heads, H, head_dim]
        weights: [B, N, heads, H, r]
        output:  [B, N, heads, r, head_dim]
        """
        projection_weights = F.softmax(mlp(src), dim=-1)
        return torch.einsum(
            "bnhed,bnher->bnhrd",
            src,
            projection_weights,
        )

    @staticmethod
    def scale_scores(scores: torch.Tensor, scale_dim: int) -> torch.Tensor:
        if scale_dim <= 0:
            raise ValueError(f"scale_dim must be positive, but got {scale_dim}.")
        return scores / (float(scale_dim) ** 0.5)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, heads, H, head_dim]
        -> [B, N, H, D]
        """
        B, nvars, heads, H, head_dim = x.shape

        x = rearrange(
            x.reshape(B * nvars, heads * H, head_dim),
            "bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d",
            hl1=heads // self.merge_size,
            hl2=H,
            hl3=self.merge_size,
        )

        return x.reshape(B, nvars, H, heads * head_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if src.ndim != 4:
            raise ValueError(f"Expected [B,N,H,D], got {src.shape}.")

        B, nvars, axis_length, D = src.shape
        if D != self.d_model:
            raise ValueError(
                f"Expected embedding dimension {self.d_model}, but got {D}."
            )
        if self.over_hidden and axis_length != self.num_channels:
            raise ValueError(
                f"Channel attention expects C={self.num_channels}, "
                f"but got axis length {axis_length}."
            )

        qkv = self.qkv(src).reshape(
            B,
            nvars,
            axis_length,
            3,
            self.n_heads,
            self.head_dim,
        )
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ----- CARD attention along the active axis -----
        if not self.over_hidden:
            # Token core: full T x T attention.
            attn_score_along_axis = torch.einsum(
                "bnhed,bnhfd->bnhef",
                self.ema(q),
                self.ema(k),
            )
            attn_score_along_axis = self.scale_scores(
                attn_score_along_axis,
                self.head_dim,
            )

            attn_along_axis = self.attn_dropout(
                F.softmax(attn_score_along_axis, dim=-1)
            )
            output_along_axis = torch.einsum(
                "bnhef,bnhfd->bnhed",
                attn_along_axis,
                v,
            )
        else:
            # Channel core: dynamic projection C -> r, then C x r attention.
            k_dp = self.dynamic_projection(k, self.dp_k)
            v_dp = self.dynamic_projection(v, self.dp_v)

            attn_score_along_axis = torch.einsum(
                "bnhed,bnhrd->bnher",
                self.ema(q),
                self.ema(k_dp),
            )
            attn_score_along_axis = self.scale_scores(
                attn_score_along_axis,
                self.head_dim,
            )

            attn_along_axis = self.attn_dropout(
                F.softmax(attn_score_along_axis, dim=-1)
            )
            output_along_axis = torch.einsum(
                "bnher,bnhrd->bnhed",
                attn_along_axis,
                v_dp,
            )

        # ----- CARD hidden-dimension attention: head_dim x head_dim -----
        attn_score_along_hidden = torch.einsum(
            "bnhae,bnhaf->bnhef",
            q,
            k,
        )
        attn_score_along_hidden = self.scale_scores(
            attn_score_along_hidden,
            axis_length,
        )

        attn_along_hidden = self.attn_dropout(
            F.softmax(attn_score_along_hidden, dim=-1)
        )
        output_along_hidden = torch.einsum(
            "bnhef,bnhaf->bnhae",
            attn_along_hidden,
            v,
        )

        # No CARD FFNs / norms / residuals here.  The two attention branches
        # are combined directly after head merging.
        output_axis = self.merge_heads(output_along_axis)
        output_hidden = self.merge_heads(output_along_hidden)

        return output_axis + output_hidden


class CARDPureAttention(nn.Module):
    """
    Pure CARD attention operator with the same external interface as the
    attention sublayer used by original DyGFormer.

    Input/output:
        [B, T, C*D]

    Internal flow:
        [B,T,C*D]
            -> reshape [B,T,C,D]
            -> pure Channel CARD attention
            -> pure Token CARD attention
            -> Channel/Token output fusion
            -> flatten [B,T,C*D]
            -> output projection

    There is intentionally NO FFN, NO LayerNorm/BatchNorm and NO residual
    connection inside this module.  Those are supplied by the unchanged
    original DyGFormer TransformerEncoder.
    """

    def __init__(
        self,
        attention_dim: int,
        channel_embedding_dim: int,
        num_heads: int,
        num_channels: int,
        dropout: float = 0.1,
        dp_rank: int = 16,
        alpha: float = 0.5,
        merge_size: int = 1,
        max_token_num: int = 128,
    ):
        super().__init__()

        if attention_dim != num_channels * channel_embedding_dim:
            raise ValueError(
                "attention_dim must equal num_channels * channel_embedding_dim, "
                f"but got {attention_dim} != {num_channels} * "
                f"{channel_embedding_dim}."
            )

        self.attention_dim = attention_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_channels = num_channels

        self.channel_attention = CARDPureAxisAttention(
            d_model=channel_embedding_dim,
            n_heads=num_heads,
            num_channels=num_channels,
            max_token_num=max_token_num,
            over_hidden=True,
            dp_rank=dp_rank,
            alpha=alpha,
            merge_size=merge_size,
            dropout=dropout,
        )

        self.token_attention = CARDPureAxisAttention(
            d_model=channel_embedding_dim,
            n_heads=num_heads,
            num_channels=num_channels,
            max_token_num=max_token_num,
            over_hidden=False,
            dp_rank=dp_rank,
            alpha=alpha,
            merge_size=merge_size,
            dropout=dropout,
        )

        # Corresponds to the output projection role of standard MHA.  Fusion
        # is performed after the two pure CARD stages and before returning to
        # the original DyGFormer residual path.
        self.out_proj = nn.Linear(attention_dim, attention_dim, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected inputs [B,T,{self.attention_dim}], got {inputs.shape}."
            )

        B, T, E = inputs.shape
        if E != self.attention_dim:
            raise ValueError(
                f"Expected attention dimension {self.attention_dim}, but got {E}."
            )

        # Recover the four DyGFormer feature channels.
        x = inputs.reshape(
            B,
            T,
            self.num_channels,
            self.channel_embedding_dim,
        )

        # Stage 1: pure channel CARD attention on [B,T,C,D].
        output_channel = self.channel_attention(x)

        # Stage 2: pure token CARD attention on [B,C,T,D].
        # There is no internal residual/normalization/FFN here.
        output_token = self.token_attention(
            output_channel.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        # CARD Channel/Token fusion, then restore the original DyGFormer
        # attention-vector layout.
        outputs = output_channel + output_token
        outputs = outputs.reshape(B, T, self.attention_dim)
        outputs = self.out_proj(outputs)

        return outputs


class TransformerEncoder(nn.Module):
    """
    Original DyGFormer TransformerEncoder with ONLY its attention operator
    replaced by CARDPureAttention.

    Input/output:
        [B, T, attention_dim] where attention_dim = C * D

    Architecture kept from original DyGFormer:
        1. Pre-LayerNorm
        2. Attention
        3. Dropout + residual
        4. Pre-LayerNorm
        5. FFN: E -> 4E -> E
        6. Dropout + residual

    CARDPureAttention itself contains no FFN, norm, or residual connection.
    """

    def __init__(
        self,
        attention_dim: int,
        channel_embedding_dim: int,
        num_heads: int,
        num_channels: int,
        dropout: float = 0.1,
        dp_rank: int = 16,
        alpha: float = 0.5,
        merge_size: int = 1,
        max_token_num: int = 128,
    ):
        super().__init__()

        self.attention_dim = attention_dim

        # The only architectural replacement relative to original DyGFormer:
        # PyTorch MultiheadAttention -> pure CARD attention.
        self.card_attention = CARDPureAttention(
            attention_dim=attention_dim,
            channel_embedding_dim=channel_embedding_dim,
            num_heads=num_heads,
            num_channels=num_channels,
            dropout=dropout,
            dp_rank=dp_rank,
            alpha=alpha,
            merge_size=merge_size,
            max_token_num=max_token_num,
        )

        self.dropout = nn.Dropout(dropout)

        # Exactly the original DyGFormer outer FFN width and layout.
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim),
        ])

        # Exactly the original DyGFormer Pre-LayerNorm structure.  Because the
        # four channels are flattened, LayerNorm jointly normalizes C * D.
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim),
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected inputs [B,T,E], got {inputs.shape}."
            )
        if inputs.shape[-1] != self.attention_dim:
            raise ValueError(
                f"Expected E={self.attention_dim}, but got {inputs.shape[-1]}."
            )

        # ----- Original DyGFormer attention sublayer -----
        # Pre-LN -> attention -> dropout -> residual.
        norm_inputs = self.norm_layers[0](inputs)
        hidden_states = self.card_attention(norm_inputs)
        outputs = inputs + self.dropout(hidden_states)

        # ----- Original DyGFormer FFN sublayer -----
        # Pre-LN -> Linear(E,4E) -> GELU -> dropout -> Linear(4E,E)
        # -> dropout -> residual.
        hidden_states = self.linear_layers[1](
            self.dropout(
                F.gelu(
                    self.linear_layers[0](
                        self.norm_layers[1](outputs)
                    )
                )
            )
        )
        outputs = outputs + self.dropout(hidden_states)

        return outputs