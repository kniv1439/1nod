import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


# ---------------------------------------------------------------------------
# CateFeatureEmbedding
# ---------------------------------------------------------------------------
class CateFeatureEmbedding(nn.Module):
    """
    将多通道离散整数特征（如 SPD 和社区图距离）嵌入为连续向量。

    输入  x: LongTensor, shape (..., num_channels)
    输出 emb: FloatTensor, shape (..., embed_dim)
    """

    def __init__(self, num_uniq_values: list, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        num_channels = len(num_uniq_values)
        # 当通道数 > 1 时，先把各通道 embedding 拼接再线性映射
        self.linear = nn.Linear(embed_dim * num_channels, embed_dim)
        num_uniq = torch.LongTensor(num_uniq_values)
        csum = torch.cumsum(num_uniq, dim=0)
        # 每个通道在统一 embedding 表中的起始 offset
        starts = torch.cat([torch.LongTensor([0]), csum[:-1]])
        self.register_buffer('num_uniq_values', num_uniq.reshape(1, 1, 1, -1))  # (1,1,1,C)
        self.register_buffer('starts', starts.reshape(1, 1, -1))                 # (1,1,C)
        self.embeddings = nn.Embedding(int(csum[-1].item()), embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor (..., num_channels)，值范围 [0, num_uniq_values[c]-1]
        """
        x = x + self.starts                   # (..., C)，offset 到统一索引
        emb = self.embeddings(x)              # (..., C, embed_dim)
        *leading, C, D = emb.shape
        emb = emb.reshape(*leading, C * D)    # (..., C*embed_dim)
        emb = self.linear(emb)               # (..., embed_dim)
        return self.layer_norm(emb)


# ---------------------------------------------------------------------------
# DyGFormerHDSE
# ---------------------------------------------------------------------------
class DyGFormerHDSE(nn.Module):

    def __init__(self,
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
                 device: str = 'cpu',
                 max_spd: int = 30,
                 max_dist: int = 10,
                 struct_embed_dim: int = 32,
                 max_temp_struct_nodes: int = 200):
        """
        DyGFormerHDSE：在 DyGFormer 的 patch 级 self-attention 上，
        通过 HDSE 方式注入拓扑结构偏置（局部原图 SPD + 局部社区图距离）。

        对每个样本，在 neighbor_sampler 的全历史邻接（nodes_neighbor_ids / times）上仅保留
        ``timestamp <= 当前交互时间 t`` 的边，
        在 src/dst 序列中出现节点的诱导子图上计算 SPD 与社区缩点图距离（规则对齐 DyGFormerST）。

        额外参数
        --------
        max_spd              : SPD 截断上限（含），embedding 表大小 = max_spd + 1。
        max_dist             : 社区图距离截断上限（含），embedding 表大小 = max_dist + 1。
        struct_embed_dim     : CateFeatureEmbedding 内部的 embedding 维度。
        max_temp_struct_nodes: 单样本诱导子图节点数上限；超过则该样本不加结构偏置（全零）。
        """
        super(DyGFormerHDSE, self).__init__()

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

        # ---- HDSE 参数 ----
        self.max_spd = max_spd
        self.max_dist = max_dist
        self.max_temp_struct_nodes = max_temp_struct_nodes
        self.use_struct_bias = True

        # 两通道的类别嵌入：[SPD(0..max_spd), dist(0..max_dist)]
        self.struct_embed = CateFeatureEmbedding(
            num_uniq_values=[max_spd + 1, max_dist + 1],
            embed_dim=struct_embed_dim,
            dropout=dropout
        )
        # (B, S, S, struct_embed_dim) → (B, S, S, num_heads)，作为 per-head 的加性偏置
        self.structure_map = nn.Linear(struct_embed_dim, num_heads)

        # ---- 原有模块 ----
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
                out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim,
                               num_heads=self.num_heads,
                               dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=self.node_feat_dim, bias=True)

    @staticmethod
    def _scalar_interact_time(t) -> float:
        if isinstance(t, torch.Tensor):
            return float(t.detach().cpu().reshape(-1)[0].item())
        if isinstance(t, np.ndarray):
            return float(np.asarray(t).reshape(-1)[0])
        return float(t)

    @staticmethod
    def _all_pairs_shortest_hops(adj: np.ndarray, max_spd: int) -> np.ndarray:
        """无权无向图全对最短路 hop，截断到 max_spd。"""
        n = adj.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.int64)
        try:
            from utils import algos
            sp, _ = algos.floyd_warshall(adj)
            sp = np.where(np.isinf(sp), max_spd, sp)
            sp = np.clip(sp, 0, max_spd).astype(np.int64)
            np.fill_diagonal(sp, 0)
            return sp
        except Exception:
            pass
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import shortest_path
            sp = shortest_path(csr_matrix(adj.astype(np.float64)), directed=False, unweighted=True)
            sp = np.where(np.isinf(sp), max_spd, sp)
            sp = np.clip(sp, 0, max_spd).astype(np.int64)
            np.fill_diagonal(sp, 0)
            return sp
        except Exception:
            pass
        dist = np.where(adj, 1.0, np.inf)
        np.fill_diagonal(dist, 0.0)
        for k in range(n):
            dist = np.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
        sp = np.where(np.isinf(dist), max_spd, dist)
        sp = np.clip(sp, 0, max_spd).astype(np.int64)
        np.fill_diagonal(sp, 0)
        return sp

    def _community_pair_distances(self, adj: np.ndarray, spd_fallback: np.ndarray) -> np.ndarray:
        """在社区缩点图上计算全对最短路 hop，失败时退化为 spd_fallback。"""
        n = adj.shape[0]
        dloc = np.full((n, n), self.max_dist, dtype=np.int64)
        np.fill_diagonal(dloc, 0)
        if n <= 1:
            return dloc
        try:
            import networkx as nx
            G = nx.from_numpy_array(adj.astype(np.int32))
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G))
            except Exception:
                communities = [{i} for i in range(n)]
            node2comm = {}
            for cid, comp in enumerate(communities):
                for u in comp:
                    node2comm[u] = cid
            K = len(communities)
            H = nx.Graph()
            H.add_nodes_from(range(K))
            for u, v in G.edges():
                cu, cv = node2comm[u], node2comm[v]
                if cu != cv:
                    H.add_edge(cu, cv)
            comm_length = dict(nx.all_pairs_shortest_path_length(H, cutoff=self.max_dist))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ci, cj = node2comm[i], node2comm[j]
                    if ci == cj:
                        dloc[i, j] = 0
                    elif ci in comm_length and cj in comm_length.get(ci, {}):
                        dloc[i, j] = min(int(comm_length[ci][cj]), self.max_dist)
            return dloc
        except Exception:
            return np.minimum(spd_fallback, self.max_dist).astype(np.int64)

    def _temporal_induced_adj(self, unique_nodes: np.ndarray, current_time: float):
        """与 DyGFormerD RWPE 一致：用 nodes_neighbor_* 中 timestamp <= current_time 的边，诱导在 unique_nodes 上。"""
        node_set = {int(x) for x in unique_nodes.flatten() if int(x) != 0}
        if not node_set:
            return np.zeros((0, 0), dtype=bool), {}
        nodes = np.array(sorted(node_set), dtype=np.int64)
        if len(nodes) > self.max_temp_struct_nodes:
            return None, None
        N = len(nodes)
        node_to_local = {int(nodes[i]): i for i in range(N)}
        adj = np.zeros((N, N), dtype=bool)
        ns = self.neighbor_sampler
        for node_id in nodes:
            nid = int(node_id)
            if nid >= len(ns.nodes_neighbor_ids):
                continue
            neighbors = ns.nodes_neighbor_ids[nid]
            ntimes = ns.nodes_neighbor_times[nid]
            if len(neighbors) == 0:
                continue
            if isinstance(ntimes, torch.Tensor):
                ntimes = ntimes.detach().cpu().numpy()
            else:
                ntimes = np.asarray(ntimes)
            neighbors = np.asarray(neighbors)
            for k in range(len(neighbors)):
                nb = int(neighbors[k])
                if nb not in node_set:
                    continue
                if float(ntimes[k]) <= current_time:
                    i, j = node_to_local[nid], node_to_local[nb]
                    adj[i, j] = True
                    adj[j, i] = True
        return adj, node_to_local

    # ------------------------------------------------------------------
    def _build_struct_attn_bias(self,
                                src_padded_nodes_neighbor_ids: np.ndarray,
                                dst_padded_nodes_neighbor_ids: np.ndarray,
                                src_num_patches: int,
                                dst_num_patches: int,
                                node_interact_times: np.ndarray) -> torch.Tensor:
        """
        按样本时间 t 在局部诱导子图上计算 SPD 与社区距离，构造 patch 级注意力加性偏置。
        """
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy()

        src_repr_np = src_padded_nodes_neighbor_ids[:, 0::self.patch_size]
        dst_repr_np = dst_padded_nodes_neighbor_ids[:, 0::self.patch_size]
        all_repr_np = np.concatenate([src_repr_np, dst_repr_np], axis=1)
        B, S = all_repr_np.shape

        spd_vals = np.zeros((B, S, S), dtype=np.int64)
        dist_vals = np.zeros((B, S, S), dtype=np.int64)

        for b in range(B):
            row_src = src_padded_nodes_neighbor_ids[b]
            row_dst = dst_padded_nodes_neighbor_ids[b]
            union = np.unique(np.concatenate([row_src[row_src != 0], row_dst[row_dst != 0]]))
            if union.size == 0:
                continue
            t = self._scalar_interact_time(node_interact_times[b])
            built = self._temporal_induced_adj(union, t)
            if built[0] is None:
                continue
            adj, node_to_local = built
            if adj.shape[0] == 0:
                continue
            spd_loc = self._all_pairs_shortest_hops(adj, self.max_spd)
            dist_loc = self._community_pair_distances(adj, spd_loc)

            repr_row = all_repr_np[b].astype(np.int64)
            loc_row = np.array([node_to_local.get(int(x), -1) for x in repr_row], dtype=np.int64)
            Li = np.tile(loc_row.reshape(S, 1), (1, S))
            Lj = np.tile(loc_row.reshape(1, S), (S, 1))
            zero_mask = (repr_row.reshape(S, 1) == 0) | (repr_row.reshape(1, S) == 0)
            valid = (Li >= 0) & (Lj >= 0) & (~zero_mask)
            spd_vals[b] = np.where(zero_mask, 0, np.where(valid, spd_loc[Li, Lj], 0))
            dist_vals[b] = np.where(zero_mask, 0, np.where(valid, dist_loc[Li, Lj], 0))

        spd_t = torch.from_numpy(spd_vals).long().to(self.device)
        dist_t = torch.from_numpy(dist_vals).long().to(self.device)
        struct_feat = torch.stack([spd_t, dist_t], dim=-1)
        struct_emb = self.struct_embed(struct_feat)
        bias = self.structure_map(struct_emb)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.num_heads, S, S)
        return bias

    # ------------------------------------------------------------------
    def compute_src_dst_node_temporal_embeddings(self,
                                                  src_node_ids: np.ndarray,
                                                  dst_node_ids: np.ndarray,
                                                  node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        """
        # 获取一阶邻居
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # padding
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids,
                               node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list,
                               nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size,
                               max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids,
                               node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list,
                               nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size,
                               max_input_sequence_length=self.max_input_sequence_length)

        # 共现特征
        src_padded_nodes_neighbor_co_occurrence_features, \
        dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # 原始特征
        src_padded_nodes_neighbor_node_raw_features, \
        src_padded_nodes_edge_raw_features, \
        src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder)

        dst_padded_nodes_neighbor_node_raw_features, \
        dst_padded_nodes_edge_raw_features, \
        dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder)

        # patches
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size)

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size)

        # 投影对齐维度
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features          = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features     = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features          = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features     = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size     = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # 拼接 src 与 dst 的 patch 序列: (B, S, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat(
            [src_patches_nodes_neighbor_node_raw_features,
             dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat(
            [src_patches_nodes_edge_raw_features,
             dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat(
            [src_patches_nodes_neighbor_time_features,
             dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat(
            [src_patches_nodes_neighbor_co_occurrence_features,
             dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        patches_data = [patches_nodes_neighbor_node_raw_features,
                        patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features,
                        patches_nodes_neighbor_co_occurrence_features]
        # (B, S, num_channels, channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)
        # (B, S, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.num_channels * self.channel_embedding_dim)

        # ---- 构造 HDSE 结构偏置 ----
        # attn_bias: (B * num_heads, S, S)
        attn_bias = self._build_struct_attn_bias(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            src_num_patches=src_num_patches,
            dst_num_patches=dst_num_patches,
            node_interact_times=node_interact_times)

        # ---- Transformer 前向（每层传入结构偏置）----
        for transformer in self.transformers:
            patches_data = transformer(patches_data, attn_bias=attn_bias)

        # 分离 src / dst
        src_patches_data = patches_data[:, :src_num_patches, :]
        dst_patches_data = patches_data[:, src_num_patches:src_num_patches + dst_num_patches, :]

        # 池化
        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    # ------------------------------------------------------------------
    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                      nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1,
                      max_input_sequence_length: int = 256):
        """pad the sequences for nodes in node_ids"""
        assert max_input_sequence_length - 1 > 0, \
            'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            assert (len(nodes_neighbor_ids_list[idx]) ==
                    len(nodes_edge_ids_list[idx]) ==
                    len(nodes_neighbor_times_list[idx]))
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx]   = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx]        = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx]  = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        padded_nodes_neighbor_ids    = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids        = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times  = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0]   = node_ids[idx]
            padded_nodes_edge_ids[idx, 0]        = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1:len(nodes_neighbor_ids_list[idx]) + 1]  = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1:len(nodes_edge_ids_list[idx]) + 1]           = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1:len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    # ------------------------------------------------------------------
    def get_features(self, node_interact_times: np.ndarray,
                     padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray,
                     time_encoder: TimeEncoder):
        """get node, edge and time features"""
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()

        padded_nodes_neighbor_node_raw_features = self.node_raw_features[
            torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[
            torch.from_numpy(padded_nodes_edge_ids)]
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(
                node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
            ).float().to(self.device))

        padded_nodes_neighbor_time_features[
            torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return (padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features)

    # ------------------------------------------------------------------
    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor,
                    padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor,
                    padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None,
                    patch_size: int = 1):
        """get the sequence of patches for nodes"""
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        patches_nodes_neighbor_node_raw_features = []
        patches_nodes_edge_raw_features = []
        patches_nodes_neighbor_time_features = []
        patches_nodes_neighbor_co_occurrence_features = []

        for patch_id in range(num_patches):
            s = patch_id * patch_size
            e = s + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, s:e, :])
            patches_nodes_edge_raw_features.append(
                padded_nodes_edge_raw_features[:, s:e, :])
            patches_nodes_neighbor_time_features.append(
                padded_nodes_neighbor_time_features[:, s:e, :])
            patches_nodes_neighbor_co_occurrence_features.append(
                padded_nodes_neighbor_co_occurrence_features[:, s:e, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        patches_nodes_neighbor_node_raw_features = torch.stack(
            patches_nodes_neighbor_node_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.node_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(
            patches_nodes_edge_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.edge_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(
            patches_nodes_neighbor_time_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.time_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(
            patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return (patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features)

    # ------------------------------------------------------------------
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


# ---------------------------------------------------------------------------
# NeighborCooccurrenceEncoder（原版保持不变）
# ---------------------------------------------------------------------------
class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim,
                      out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray,
                                dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        for src_ids, dst_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):
            src_unique_keys, src_inverse_indices, src_counts = np.unique(
                src_ids, return_inverse=True, return_counts=True)
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(
                src_counts[src_inverse_indices]).float().to(self.device)
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                dst_ids, return_inverse=True, return_counts=True)
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                dst_counts[dst_inverse_indices]).float().to(self.device)
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                src_ids.copy()).apply_(
                lambda nid: dst_mapping_dict.get(nid, 0.0)).float().to(self.device)
            src_padded_nodes_appearances.append(
                torch.stack([src_padded_node_neighbor_counts_in_src,
                             src_padded_node_neighbor_counts_in_dst], dim=1))

            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(
                dst_ids.copy()).apply_(
                lambda nid: src_mapping_dict.get(nid, 0.0)).float().to(self.device)
            dst_padded_nodes_appearances.append(
                torch.stack([dst_padded_node_neighbor_counts_in_src,
                             dst_padded_node_neighbor_counts_in_dst], dim=1))

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0
        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray,
                dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = \
            self.count_nodes_appearances(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)
        src_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encode_layer(
                src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encode_layer(
                dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        return (src_padded_nodes_neighbor_co_occurrence_features,
                dst_padded_nodes_neighbor_co_occurrence_features)


# ---------------------------------------------------------------------------
# TransformerEncoder — 修改：接受并应用 attn_bias
# ---------------------------------------------------------------------------
class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder with optional additive structural attention bias.

        参数
        ----
        attention_dim : 输入/输出维度
        num_heads     : 注意力头数
        dropout       : dropout 概率
        """
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor,
                attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        参数
        ----
        inputs    : (batch_size, num_patches, attention_dim)
        attn_bias : (batch_size * num_heads, num_patches, num_patches) float 或 None.
                    在 softmax 前以加性方式注入到 QK^T/sqrt(d) 上，
                    等价于  logits_{ij} += attn_bias_{ij}。

        返回
        ----
        outputs : (batch_size, num_patches, attention_dim)
        """
        # MultiheadAttention 期望 (seq_len, batch, dim)
        transposed_inputs = inputs.transpose(0, 1)       # (S, B, D)
        transposed_inputs = self.norm_layers[0](transposed_inputs)

        # attn_mask 为 float 时是加性偏置，在 softmax 前叠加：
        #   logits = QK^T/sqrt(d) + attn_mask
        # 支持 (S, S) 或 (B*H, S, S)，此处传 (B*H, S, S)
        hidden_states = self.multi_head_attention(
            query=transposed_inputs,
            key=transposed_inputs,
            value=transposed_inputs,
            attn_mask=attn_bias       # (B*num_heads, S, S) float
        )[0].transpose(0, 1)          # 还原成 (B, S, D)

        outputs = inputs + self.dropout(hidden_states)
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        return outputs + self.dropout(hidden_states)
