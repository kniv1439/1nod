import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.nn import MultiheadAttention
from scipy.sparse import csr_matrix

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


def _lookup_counts_per_key(unique_keys: np.ndarray, counts: np.ndarray, query: np.ndarray) -> np.ndarray:
    """For each value in query, return its total count under (unique_keys, counts).

    unique_keys must be sorted ascending (as from np.unique). Keys absent from unique_keys map to 0.
    """
    if unique_keys.size == 0:
        return np.zeros(query.shape, dtype=np.float64)
    idx = np.searchsorted(unique_keys, query, side="left")
    idx_clip = np.clip(idx, 0, unique_keys.size - 1)
    hit = (idx < unique_keys.size) & (unique_keys[idx_clip] == query)
    out = np.zeros(query.shape, dtype=np.float64)
    out[hit] = counts[idx_clip[hit]].astype(np.float64, copy=False)
    return out


class DyGFormerSPSE(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu',
                 spse_dist_len: int = 2,
                 spse_root_nodes_fact: float = 0.075,
                 spse_min_root_nodes: int = 4,
                 spse_max_root_nodes: int = 0,
                 spse_max_bfs_depth: int = 0,
                 spse_max_tries_per_depth: int = 1,
                 spse_use_adjacency_walk_counts: bool = True,
                 spse_max_subgraph_nodes: int = 200):
        """
        DyGFormer model with SPSE structural bias on patch-patch attention.

        这版优化点：
        1. 仅用 patch representative nodes（每个 patch 第一个节点）构子图
        2. 每个节点的邻居列表按时间升序预排序
        3. 每次对 sample 时间 t 用 np.searchsorted 找 cutoff，只取 < t 的邻居前缀
        4. 用 set/dict 做 membership 与 global->local 映射
        """
        super(DyGFormerSPSE, self).__init__()

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

        self.spse_dist_len = spse_dist_len
        self.spse_root_nodes_fact = spse_root_nodes_fact
        self.spse_min_root_nodes = spse_min_root_nodes
        self.spse_max_root_nodes = spse_max_root_nodes
        self.spse_max_bfs_depth = spse_max_bfs_depth
        self.spse_max_tries_per_depth = spse_max_tries_per_depth
        self.spse_use_adjacency_walk_counts = spse_use_adjacency_walk_counts
        self.spse_max_subgraph_nodes = spse_max_subgraph_nodes
        self.spse_bias_feat_dim = 2 if spse_use_adjacency_walk_counts else spse_dist_len
        self.spse_linear = nn.Linear(self.spse_bias_feat_dim, 1, bias=True)

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4

        self.transformers = nn.ModuleList([
            TransformerEncoder(
                attention_dim=self.num_channels * self.channel_embedding_dim,
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

        self._preconvert_neighbor_lists()
        self._get_simple_paths_count = None
        if not self.spse_use_adjacency_walk_counts:
            self._load_spse_module()

    def _preconvert_neighbor_lists(self):
        """Pre-convert neighbor id/time lists to numpy arrays and sort them by time ascending."""
        ns = self.neighbor_sampler

        self._ns_ids_np = []
        self._ns_times_np = []

        for ids_arr, times_arr in zip(ns.nodes_neighbor_ids, ns.nodes_neighbor_times):
            ids_np = ids_arr.cpu().numpy() if isinstance(ids_arr, torch.Tensor) else np.asarray(ids_arr)
            times_np = times_arr.cpu().numpy() if isinstance(times_arr, torch.Tensor) else np.asarray(times_arr)

            ids_np = np.asarray(ids_np)
            times_np = np.asarray(times_np)

            if len(ids_np) > 1:
                # 确保按时间升序，后续才能 searchsorted 截断
                order = np.argsort(times_np, kind='stable')
                ids_np = ids_np[order]
                times_np = times_np[order]

            self._ns_ids_np.append(ids_np)
            self._ns_times_np.append(times_np)

        self._ns_deg_np = np.fromiter(
            (len(x) for x in self._ns_ids_np), dtype=np.int64, count=len(self._ns_ids_np)
        )

    def _load_spse_module(self):
        """Import count_self_avoiding_paths once to avoid thread-unsafe sys.path mutation."""
        import sys
        _spse_dir = os.path.join(os.path.dirname(__file__), '..', 'Graph-SPSE-Encoding')
        if _spse_dir not in sys.path:
            sys.path.insert(0, _spse_dir)
        from count_self_avoiding_paths import get_simple_paths_count
        self._get_simple_paths_count = get_simple_paths_count

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        """
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, node_interact_times=node_interact_times
            )

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids, node_interact_times=node_interact_times
            )

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

        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids
            )

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder
            )

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder
            )

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

        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        patches_nodes_neighbor_node_raw_features = torch.cat(
            [src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat(
            [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat(
            [src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat(
            [src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        patches_data = [patches_nodes_neighbor_node_raw_features,
                        patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features,
                        patches_nodes_neighbor_co_occurrence_features]
        patches_data = torch.stack(patches_data, dim=2)
        patches_data = patches_data.reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.num_channels * self.channel_embedding_dim
        )

        struct_bias = self.compute_spse_bias(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            node_interact_times=node_interact_times,
            src_num_patches=src_num_patches,
            dst_num_patches=dst_num_patches,
        )

        for transformer in self.transformers:
            patches_data = transformer(patches_data, struct_bias=struct_bias)

        src_patches_data = patches_data[:, : src_num_patches, :]
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
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

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
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

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
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
            patches_nodes_neighbor_co_occurrence_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return (patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features)

    def compute_spse_bias(self,
                          src_padded_nodes_neighbor_ids: np.ndarray,
                          dst_padded_nodes_neighbor_ids: np.ndarray,
                          node_interact_times: np.ndarray,
                          src_num_patches: int,
                          dst_num_patches: int) -> torch.Tensor:
        """
        Compute the SPSE-based structural bias matrix for patch-patch attention.

        这版优化：
        - all_ids 只来自 patch representative nodes
        - 每个节点邻居列表按时间升序
        - 用 searchsorted 截断 < t 的历史邻居前缀
        - 用 set/dict 做 membership 与映射
        """
        dev = self.device
        B = len(node_interact_times)
        L = src_num_patches + dst_num_patches
        K = self.spse_bias_feat_dim

        c_spse_cpu = np.zeros((B, L, L, K), dtype=np.float32)

        ns_ids = self._ns_ids_np
        ns_times = self._ns_times_np
        ns_deg = self._ns_deg_np
        n_total = len(ns_ids)

        patch_size = self.patch_size
        max_sub_nodes = self.spse_max_subgraph_nodes
        use_walk_counts = self.spse_use_adjacency_walk_counts

        spse_min_root = self.spse_min_root_nodes
        spse_max_root = self.spse_max_root_nodes
        spse_root_fact = self.spse_root_nodes_fact
        spse_dist_len = self.spse_dist_len
        spse_bfs_depth = self.spse_max_bfs_depth
        spse_tries = self.spse_max_tries_per_depth
        get_spc = self._get_simple_paths_count

        src_patch_starts = np.arange(src_num_patches) * patch_size
        dst_patch_starts = np.arange(dst_num_patches) * patch_size

        def _process_sample(b: int):
            t = float(node_interact_times[b])

            # 1) patch representative nodes
            src_rep_gids = src_padded_nodes_neighbor_ids[b, src_patch_starts].astype(np.int64)
            dst_rep_gids = dst_padded_nodes_neighbor_ids[b, dst_patch_starts].astype(np.int64)
            rep_gids = np.concatenate([src_rep_gids, dst_rep_gids]).astype(np.int64)

            all_ids = np.unique(rep_gids)
            all_ids = all_ids[all_ids > 0]

            if len(all_ids) < 2:
                return
            if max_sub_nodes > 0 and len(all_ids) > max_sub_nodes:
                return

            # 哈希 membership / local map
            all_ids_list = all_ids.tolist()
            all_id_set = set(all_ids_list)
            local_map = {gid: i for i, gid in enumerate(all_ids_list)}

            valid_us = all_ids[all_ids < n_total].astype(np.int64)
            has_nbr = ns_deg[valid_us] > 0
            valid_us_nbr = valid_us[has_nbr]

            if len(valid_us_nbr) == 0:
                return

            # 2) 对每个 u 单独 searchsorted 截断历史邻居前缀
            edge_set = set()
            for u in valid_us_nbr.tolist():
                nbr_ids_u = ns_ids[u]
                nbr_times_u = ns_times[u]

                if len(nbr_ids_u) == 0:
                    continue

                cutoff = np.searchsorted(nbr_times_u, t, side="left")
                if cutoff <= 0:
                    continue

                hist_nbr_ids = nbr_ids_u[:cutoff]

                # 3) membership test: 只保留端点也在 all_ids 的边
                u_local = local_map[u]
                for v in hist_nbr_ids.tolist():
                    if v == u:
                        continue
                    if v in all_id_set:
                        v_local = local_map[v]
                        a = u_local if u_local < v_local else v_local
                        c = v_local if u_local < v_local else u_local
                        edge_set.add((a, c))

            if len(edge_set) == 0:
                return

            # 4) 无向局部图
            edge_list = np.array(list(edge_set), dtype=np.int64)
            dedup_src = edge_list[:, 0]
            dedup_dst = edge_list[:, 1]

            N_prime = len(all_ids)
            ei_np = np.stack([dedup_src, dedup_dst], axis=0)

            deg = np.bincount(ei_np.ravel(), minlength=N_prime)
            core_indices = np.where(deg > 0)[0]
            N_eff = len(core_indices)
            if N_eff < 2:
                return

            old_to_new = np.full(N_prime, -1, dtype=np.int64)
            old_to_new[core_indices] = np.arange(N_eff, dtype=np.int64)

            src_e = old_to_new[ei_np[0]]
            dst_e = old_to_new[ei_np[1]]

            # 5) patch reps -> local core index
            rep_local_full = np.full(rep_gids.shape, -1, dtype=np.int64)
            for i, gid in enumerate(rep_gids.tolist()):
                if gid > 0 and gid in local_map:
                    rep_local_full[i] = local_map[gid]

            safe_full = np.clip(rep_local_full, 0, N_prime - 1)
            rep_local = np.where(rep_local_full >= 0, old_to_new[safe_full], -1)

            valid_mask_l = rep_local >= 0
            if valid_mask_l.sum() < 2:
                return

            patch_idx_np = np.where(valid_mask_l)[0]
            local_idx_np = rep_local[valid_mask_l]
            len_valid = len(local_idx_np)

            try:
                if use_walk_counts:
                    local_to_row = np.full(N_eff, -1, dtype=np.int64)
                    local_to_row[local_idx_np] = np.arange(len_valid, dtype=np.int64)

                    # 1-hop among valid patch reps
                    A_sub = np.zeros((len_valid, len_valid), dtype=np.float32)

                    pu = local_to_row[src_e]
                    pv = local_to_row[dst_e]
                    m = (pu >= 0) & (pv >= 0)
                    if np.any(m):
                        pui = pu[m]
                        pvi = pv[m]
                        A_sub[pui, pvi] = 1.0
                        A_sub[pvi, pui] = 1.0

                    # 稀疏矩阵求 2-hop
                    row_mask = local_to_row[src_e] >= 0
                    row_i = local_to_row[src_e[row_mask]]
                    col_j = dst_e[row_mask]

                    if len(row_i) == 0:
                        return

                    A_sp = csr_matrix(
                        (np.ones(len(row_i), dtype=np.float32), (row_i, col_j)),
                        shape=(len_valid, N_eff)
                    )
                    A2_sub = A_sp.dot(A_sp.transpose()).toarray()

                    M_sub_np = np.stack([A_sub, A2_sub], axis=0)
                else:
                    edge_index_eff = torch.from_numpy(np.stack([src_e, dst_e])).long()
                    n_root = max(spse_min_root, int(N_eff * spse_root_fact))
                    if spse_max_root > 0:
                        n_root = min(n_root, spse_max_root)

                    M = get_spc(
                        edge_index_eff,
                        dist_len=spse_dist_len,
                        n_root_nodes=n_root,
                        max_bfs_depth=spse_bfs_depth,
                        max_tries_per_depth=spse_tries,
                        n_reps=1,
                        directed=False,
                        allow_cycles=False,
                        max_num_nodes=N_eff,
                    )
                    local_idx_t = torch.from_numpy(local_idx_np).long()
                    M_sub_np = M.squeeze(0).float()[:, local_idx_t][:, :, local_idx_t].numpy()
            except Exception:
                return

            sub_M_np = M_sub_np.transpose(1, 2, 0)
            c_spse_cpu[b,
                       patch_idx_np[:, np.newaxis],
                       patch_idx_np[np.newaxis, :], :] = sub_M_np

        n_workers = min(B, os.cpu_count() or 4)
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(_process_sample, range(B)))
        else:
            for b in range(B):
                _process_sample(b)

        c_spse = torch.from_numpy(c_spse_cpu).to(dev)
        c_spse = torch.log1p(c_spse)
        bias = self.spse_linear(c_spse).squeeze(-1)

        return bias

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
        self._preconvert_neighbor_lists()


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

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        """
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []

        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids,
                                                                              dst_padded_nodes_neighbor_ids):

            src_unique_keys, src_inverse_indices, src_counts = np.unique(
                src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(
                src_counts[src_inverse_indices]).float().to(self.device)

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                dst_counts[dst_inverse_indices]).float().to(self.device)

            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(
                _lookup_counts_per_key(dst_unique_keys, dst_counts, src_padded_node_neighbor_ids)
            ).float().to(self.device)
            src_padded_nodes_appearances.append(
                torch.stack([src_padded_node_neighbor_counts_in_src,
                             src_padded_node_neighbor_counts_in_dst], dim=1)
            )

            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(
                _lookup_counts_per_key(src_unique_keys, src_counts, dst_padded_node_neighbor_ids)
            ).float().to(self.device)
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
        compute the neighbor co-occurrence features
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

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        """
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor, struct_bias: torch.Tensor = None):
        """
        encode the inputs by Transformer encoder
        """
        B, L, _ = inputs.shape
        transposed_inputs = inputs.transpose(0, 1)
        transposed_inputs = self.norm_layers[0](transposed_inputs)

        attn_mask = None
        if struct_bias is not None:
            attn_mask = struct_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(
                B * self.num_heads, L, L
            )

        hidden_states = self.multi_head_attention(
            query=transposed_inputs,
            key=transposed_inputs,
            value=transposed_inputs,
            attn_mask=attn_mask
        )[0].transpose(0, 1)

        outputs = inputs + self.dropout(hidden_states)
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs))))
        )
        outputs = outputs + self.dropout(hidden_states)
        return outputs