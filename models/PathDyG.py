import numpy as np  # type: ignore
import torch
import torch.nn as nn
import sys

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.AMS import AMS  # type: ignore  # AMS模块（Pathformer核心）
from utils.RevIN1 import RevIN  # type: ignore  # RevIN模块
from utils.Embedding import PositionalEmbedding, TokenEmbedding  # type: ignore  # 嵌入模块
from utils.Other import FourierLayer, series_decomp_multi  # type: ignore  # 分解/频域模块


class PathDyG(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        channel_embedding_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_input_sequence_length: int = 32,
        device: str = 'cpu',
        num_heads: int = 8,
    ):
        """
        PathDyG model (带 Pathformer-style 时序建模).

        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler
        :param time_feat_dim: 维度 of time encoding
        :param channel_embedding_dim: d_model for sequence backbone
        :param num_layers: AMS 层数
        :param dropout: dropout
        :param max_input_sequence_length: 序列最大长度
        :param device: 设备
        :param num_heads: attention heads 数（目前没直接用在 AMS 里，可留作后续扩展）
        """
        super(PathDyG, self).__init__()

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)
        ).to(device)

        # 多尺度分解模块（类似 Pathformer Router 中的 trend/seasonal）
        self.decomp = series_decomp_multi(kernel_size=[25])

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        self.num_heads = num_heads

        # 特征缩放因子，用于平衡不同通道的贡献
        self.feature_scale_factors = {
            'node': 1.0,
            'edge': 1.0,
            'time': 1.0,
            'neighbor_co_occurrence': 1.0,
        }

        # 时间编码器
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # 邻居共现编码器
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device,
        )

        # 特征投影层 - 固定4通道
        self.projection_layer = nn.ModuleDict(
            {
                'node': nn.Linear(
                    in_features=self.node_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                'edge': nn.Linear(
                    in_features=self.edge_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                'time': nn.Linear(
                    in_features=self.time_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                'neighbor_co_occurrence': nn.Linear(
                    in_features=self.neighbor_co_occurrence_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
            }
        )

        # 嵌入模块：4 * C_in -> d_model (channel_embedding_dim)
        # [B, L, 4*C] -> TokenEmbedding -> [B, L, C]
        input_dim = 4 * self.channel_embedding_dim
        self.token_embedding = TokenEmbedding(
            c_in=input_dim, d_model=self.channel_embedding_dim
        )
        self.positional_embedding = PositionalEmbedding(
            d_model=self.channel_embedding_dim
        )

        # RevIN归一化层（对 [B, L, C] 做 Instance-like 归一化）
        self.revin = RevIN(
            num_features=self.channel_embedding_dim,
            affine=True,
            subtract_last=False,
        )

        # Fourier层（频域处理）
        # pred_len 设为 max_input_sequence_length，使得输入输出长度一致
        self.fourier_layer = FourierLayer(
            pred_len=self.max_input_sequence_length,
            k=4,
            low_freq=1,
            output_attention=False,
        )

        # 定义每个 AMS 层的 patch_size 列表
        patch_size_list = [8, 6, 4, 2]
        num_experts = len(patch_size_list)

        # AMS层（Pathformer核心，多尺度 patch + dual attention + router）
        self.ams_layers = nn.ModuleList(
            [
                AMS(
                    input_size=self.channel_embedding_dim,
                    output_size=self.channel_embedding_dim,
                    num_experts=num_experts,
                    device=device,
                    num_nodes=1,
                    d_model=self.channel_embedding_dim,
                    d_ff=self.channel_embedding_dim * 4,
                    patch_size=patch_size_list,  # 多尺度 patch 列表
                    noisy_gating=True,
                    k=min(4, num_experts),      # top-k 路由
                    layer_number=i + 1,
                    residual_connection=True,
                    batch_norm=True,
                )
                for i in range(self.num_layers)
            ]
        )

        # 输出层：把时间维聚合后的表示映射回 node_feat_dim
        self.output_layer = nn.Linear(
            in_features=self.channel_embedding_dim,
            out_features=self.node_feat_dim,
            bias=True,
        )

    # ----------- 小工具：兼容 RevIN / FourierLayer / decomp 的不同实现 -----------

    def _apply_revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        兼容不同 RevIN 实现的 forward 签名：
        - RevIN(x)
        - RevIN(x, 'norm')
        """
        try:
            return self.revin(x)
        except TypeError:
            return self.revin(x, 'norm')

    def _apply_fourier(self, x: torch.Tensor) -> torch.Tensor:
        """
        兼容 FourierLayer 返回 x 或 (x, attn) 的情况
        """
        out = self.fourier_layer(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def _apply_decomp(self, x: torch.Tensor):
        """
        兼容 series_decomp_multi 返回 (seasonal, trend) 或 (trend, seasonal)
        这里假设返回 (seasonal, trend)，如果你本地实现反了可以自己调换。
        """
        out = self.decomp(x)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            seasonal, trend = out
        else:
            # 保底：没有 decomposition 就全当 seasonal 用
            seasonal, trend = x, torch.zeros_like(x)
        return seasonal, trend

    # --------------------------------------------------------------------------

    def forward(
        self,
        batch=None,
        src_node_ids=None,
        dst_node_ids=None,
        node_interact_times=None,
        **kwargs
    ):
        """
        兼容两种调用方式：
        1. forward(batch) - batch 是包含 src_node_ids, dst_node_ids, node_interact_times 的字典
        2. forward(src_node_ids=..., dst_node_ids=..., node_interact_times=...)
        """
        if batch is not None:
            src_node_ids = batch['src_node_ids']
            dst_node_ids = batch['dst_node_ids']
            node_interact_times = batch['node_interact_times']

        return self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times
        )

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = None,
        **kwargs
    ):
        """
        Compute source and destination node temporal embeddings.
        最终返回:
            src_out: [B, node_feat_dim]
            dst_out: [B, node_feat_dim]
            None   : 与其他模型接口对齐
        """

        # 1. 获取源、目标节点的一阶邻居（序列）
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times
            )

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times
            )

        # 2. 序列 padding
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                nodes_edge_ids_list=src_nodes_edge_ids_list,
                nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                max_input_sequence_length=self.max_input_sequence_length
            )

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                nodes_edge_ids_list=dst_nodes_edge_ids_list,
                nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                max_input_sequence_length=self.max_input_sequence_length
            )

        # 3. 取 node / edge / time 原始特征
        src_node_raw, src_edge_raw, src_time_feat = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            padded_nodes_edge_ids=src_padded_nodes_edge_ids,
            padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            time_encoder=self.time_encoder,
        )

        dst_node_raw, dst_edge_raw, dst_time_feat = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
            padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
            time_encoder=self.time_encoder,
        )

        # 4. 计算邻居共现特征
        src_neighbor_cooc, dst_neighbor_cooc = self.neighbor_co_occurrence_encoder(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
        )

        # 5. 各通道线性投影到 channel_embedding_dim
        feature_list_src = [
            self.feature_scale_factors['node']
            * self.projection_layer['node'](src_node_raw),
            self.feature_scale_factors['edge']
            * self.projection_layer['edge'](src_edge_raw),
            self.feature_scale_factors['time']
            * self.projection_layer['time'](src_time_feat),
            self.feature_scale_factors['neighbor_co_occurrence']
            * self.projection_layer['neighbor_co_occurrence'](src_neighbor_cooc),
        ]

        feature_list_dst = [
            self.feature_scale_factors['node']
            * self.projection_layer['node'](dst_node_raw),
            self.feature_scale_factors['edge']
            * self.projection_layer['edge'](dst_edge_raw),
            self.feature_scale_factors['time']
            * self.projection_layer['time'](dst_time_feat),
            self.feature_scale_factors['neighbor_co_occurrence']
            * self.projection_layer['neighbor_co_occurrence'](dst_neighbor_cooc),
        ]

        # [B, L, 4*C]
        src_features = torch.cat(feature_list_src, dim=-1)
        dst_features = torch.cat(feature_list_dst, dim=-1)

        # -----------------------------------------
        # 6. Pathformer-style 时序建模流水线
        #    6.1 Token + Positional Embedding
        # -----------------------------------------
        # TokenEmbedding: [B, L, 4C] -> [B, L, C]
        src_embed = self.token_embedding(src_features)
        dst_embed = self.token_embedding(dst_features)

        # PositionalEmbedding: 输入 [B, L, C]
        src_pos = self.positional_embedding(src_embed)
        dst_pos = self.positional_embedding(dst_embed)

        x_src = src_embed + src_pos
        x_dst = dst_embed + dst_pos

        # -----------------------------------------
        # 6.2 RevIN 归一化（只做 norm，不反归一化）
        # -----------------------------------------
        x_src = self._apply_revin_norm(x_src)
        x_dst = self._apply_revin_norm(x_dst)

        # -----------------------------------------
        # 6.3 时序分解 + 频域建模（可选增强）
        #     输入假定为 [B, L, C]
        # -----------------------------------------
        src_seasonal, src_trend = self._apply_decomp(x_src)
        dst_seasonal, dst_trend = self._apply_decomp(x_dst)

        x_src = x_src + src_seasonal + src_trend
        x_dst = x_dst + dst_seasonal + dst_trend

        x_src = self._apply_fourier(x_src)
        x_dst = self._apply_fourier(x_dst)

        # -----------------------------------------
        # 6.4 AMS 多尺度 Transformer Block 堆叠（核心）
        # -----------------------------------------
        for ams_layer in self.ams_layers:
            x_src, _ = ams_layer(x_src)  # 解包返回值，忽略 balance_loss
            x_dst, _ = ams_layer(x_dst)  # 解包返回值，忽略 balance_loss

        # -----------------------------------------
        # 6.5 聚合时间维 + 输出映射
        # -----------------------------------------
        # 这里选择序列的第 0 个位置（当前节点）作为最终表征
        src_final = x_src[:, 0, :]  # [B, C]
        dst_final = x_dst[:, 0, :]  # [B, C]

        src_out = self.output_layer(src_final)  # [B, node_feat_dim]
        dst_out = self.output_layer(dst_final)

        # 返回 3 个值以保持与其他模型接口一致（第 3 个值为 None）
        return src_out, dst_out, None

    # ----------------------------------------------------------------------

    def pad_sequences(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids_list: list,
        nodes_edge_ids_list: list,
        nodes_neighbor_times_list: list,
        max_input_sequence_length: int = 256,
    ):
        """
        Pad the sequences for nodes in node_ids.
        """
        max_seq_length = max_input_sequence_length
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(
            np.longlong
        )
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(
            np.longlong
        )
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(
            np.float32
        )

        for idx in range(len(node_ids)):
            # 位置 0 存当前节点 & 当前交互时间
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                max_neighbors = max_seq_length - 1  # 最大邻居数量
                nodes_neighbor_ids = nodes_neighbor_ids_list[idx][:max_neighbors]
                nodes_edge_ids = nodes_edge_ids_list[idx][:max_neighbors]
                nodes_neighbor_times = nodes_neighbor_times_list[idx][:max_neighbors]

                padded_nodes_neighbor_ids[idx, 1 : len(nodes_neighbor_ids) + 1] = (
                    nodes_neighbor_ids
                )
                padded_nodes_edge_ids[idx, 1 : len(nodes_edge_ids) + 1] = (
                    nodes_edge_ids
                )
                padded_nodes_neighbor_times[
                    idx, 1 : len(nodes_neighbor_times) + 1
                ] = nodes_neighbor_times

        return (
            padded_nodes_neighbor_ids,
            padded_nodes_edge_ids,
            padded_nodes_neighbor_times,
        )

    def get_features(
        self,
        node_interact_times: np.ndarray,
        padded_nodes_neighbor_ids: np.ndarray,
        padded_nodes_edge_ids: np.ndarray,
        padded_nodes_neighbor_times: np.ndarray,
        time_encoder: TimeEncoder,
    ):
        """
        Get node, edge, and time features.
        """
        # 保证输入都是 numpy
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()
        if isinstance(padded_nodes_neighbor_ids, torch.Tensor):
            padded_nodes_neighbor_ids = padded_nodes_neighbor_ids.cpu().numpy()
        if isinstance(padded_nodes_edge_ids, torch.Tensor):
            padded_nodes_edge_ids = padded_nodes_edge_ids.cpu().numpy()
        if isinstance(padded_nodes_neighbor_times, torch.Tensor):
            padded_nodes_neighbor_times = padded_nodes_neighbor_times.cpu().numpy()

        padded_nodes_neighbor_node_raw_features = self.node_raw_features[
            torch.from_numpy(padded_nodes_neighbor_ids)
        ]
        padded_nodes_edge_raw_features = self.edge_raw_features[
            torch.from_numpy(padded_nodes_edge_ids)
        ]

        # 时间差编码：当前交互时间 - 邻居时间
        time_deltas = (
            node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
        ).astype(np.float32)
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(time_deltas)
            .float()
            .to(self.device)
        )

        # mask 掉 padding（id == 0）的时间特征
        padded_nodes_neighbor_time_features[
            torch.from_numpy(padded_nodes_neighbor_ids == 0)
        ] = 0.0

        return (
            padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features,
        )

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        Set the neighbor sampler for the model.
        :param neighbor_sampler: NeighborSampler instance
        """
        self.neighbor_sampler = neighbor_sampler


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
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []

        for (
            src_padded_node_neighbor_ids,
            dst_padded_node_neighbor_ids,
        ) in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):

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

            # in_dst for src-seq
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                src_padded_node_neighbor_ids.copy()
            ).apply_(lambda nid: dst_mapping_dict.get(nid, 0.0)).float().to(
                self.device
            )
            src_padded_nodes_appearances.append(
                torch.stack(
                    [
                        src_padded_node_neighbor_counts_in_src,
                        src_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )

            # in_src for dst-seq
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(
                dst_padded_node_neighbor_ids.copy()
            ).apply_(lambda nid: src_mapping_dict.get(nid, 0.0)).float().to(
                self.device
            )
            dst_padded_nodes_appearances.append(
                torch.stack(
                    [
                        dst_padded_node_neighbor_counts_in_src,
                        dst_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )

        src_padded_nodes_appearances = torch.stack(
            src_padded_nodes_appearances, dim=0
        )
        dst_padded_nodes_appearances = torch.stack(
            dst_padded_nodes_appearances, dim=0
        )

        # mask 掉 padding（id == 0）
        src_padded_nodes_appearances[
            torch.from_numpy(src_padded_nodes_neighbor_ids == 0)
        ] = 0.0
        dst_padded_nodes_appearances[
            torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)
        ] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(
        self,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
    ):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        src_padded_nodes_appearances, dst_padded_nodes_appearances = (
            self.count_nodes_appearances(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            )
        )

        # encode + sum over co-occurrence type dim (=2)
        src_feats = self.neighbor_co_occurrence_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)
        ).sum(dim=2)
        dst_feats = self.neighbor_co_occurrence_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)
        ).sum(dim=2)

        return src_feats, dst_feats
