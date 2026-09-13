import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.utils import to_dense_adj

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class DyGFormerRRWP(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu',
                 rrwp_walk_length: int = 8,
                 rrwp_add_identity: bool = True):
        """
        DyGFormer model with RRWP (Relative Random Walk Probabilities) structural bias on patch-patch attention.
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
        :param rrwp_walk_length: int, number of random walk steps K.
               With add_identity=True: pe_list = [I, T, T^2, ..., T^{K-1}], giving K features per node pair.
        :param rrwp_add_identity: bool, whether to include the identity matrix (k=0 self-loop term) as the first feature.
        """
        super(DyGFormerRRWP, self).__init__()

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

        self.rrwp_walk_length = rrwp_walk_length
        self.rrwp_add_identity = rrwp_add_identity
        # With add_identity=True: [I, T, T^2, ..., T^{K-1}] → K items total
        # With add_identity=False: [T, T^2, ..., T^{K}] → K items total
        self.rrwp_feat_dim = rrwp_walk_length
        self.rrwp_linear = nn.Linear(self.rrwp_feat_dim, 1, bias=True)

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
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
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

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

        # Compute RRWP structural bias: (batch_size, L, L)
        struct_bias = self.compute_rrwp_bias(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            node_interact_times=node_interact_times,
            src_num_patches=src_num_patches,
            dst_num_patches=dst_num_patches,
        )

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            patches_data = transformer(patches_data, struct_bias=struct_bias)

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
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()

        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
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
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def compute_rrwp_bias(self,
                          src_padded_nodes_neighbor_ids: np.ndarray,
                          dst_padded_nodes_neighbor_ids: np.ndarray,
                          node_interact_times: np.ndarray,
                          src_num_patches: int,
                          dst_num_patches: int) -> torch.Tensor:
        """
        Compute RRWP-based structural bias matrix for patch-patch attention.
        For each batch element b and timestamp t = node_interact_times[b]:
          1. Collect all unique non-zero node ids from src and dst padded sequences.
          2. Build a temporal subgraph using all edges before t among these nodes.
          3. Compute row-normalised random walk matrix T = D^{-1} A, then stack
             [I, T, T^2, ..., T^{K-1}] → M of shape (K, N_eff, N_eff).
             Values are transition probabilities in [0, 1].
          4. For each patch pair (p, q), look up M[:, i_p, i_q] → K-dim vector.
          5. Apply rrwp_linear (K -> 1) to get scalar bias.

        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids: ndarray, shape (batch_size, dst_max_seq_length)
        :param node_interact_times: ndarray, shape (batch_size,)
        :param src_num_patches: int
        :param dst_num_patches: int
        :return: bias Tensor, shape (batch_size, L, L), on self.device
        """
        dev = self.device
        B = len(node_interact_times)
        L = src_num_patches + dst_num_patches
        K = self.rrwp_feat_dim

        c_rrwp = torch.zeros(B, L, L, K, dtype=torch.float32, device=dev)

        for b in range(B):
            t = float(node_interact_times[b])

            # 1. Collect unique non-zero node ids from both sequences
            src_ids = src_padded_nodes_neighbor_ids[b]
            dst_ids = dst_padded_nodes_neighbor_ids[b]
            all_ids = np.unique(np.concatenate([src_ids, dst_ids]))
            all_ids = all_ids[all_ids > 0]

            if len(all_ids) < 2:
                continue

            id_set = set(all_ids.tolist())
            id_to_local = {int(gid): lid for lid, gid in enumerate(all_ids.tolist())}
            N_prime = len(all_ids)

            # 2. Build subgraph: edges with time < t, both endpoints in id_set
            edge_src_list, edge_dst_list = [], []
            for u in all_ids:
                u_int = int(u)
                if u_int >= len(self.neighbor_sampler.nodes_neighbor_ids):
                    continue
                nbr_ids = self.neighbor_sampler.nodes_neighbor_ids[u_int]
                nbr_times = self.neighbor_sampler.nodes_neighbor_times[u_int]
                mask = nbr_times < t
                for v in nbr_ids[mask]:
                    v_int = int(v)
                    if v_int in id_set and v_int != u_int:
                        edge_src_list.append(id_to_local[u_int])
                        edge_dst_list.append(id_to_local[v_int])

            if len(edge_src_list) == 0:
                continue

            # Undirected: merge both directions and deduplicate
            src_arr = np.array(edge_src_list)
            dst_arr = np.array(edge_dst_list)
            edges = np.unique(
                np.stack([np.concatenate([src_arr, dst_arr]),
                          np.concatenate([dst_arr, src_arr])], axis=1),
                axis=0
            )
            edge_index_local = torch.from_numpy(edges.T).long()  # (2, num_edges)

            # 2b. Drop isolated nodes to shrink the dense matrix
            ei_np = edge_index_local.numpy()
            deg = np.bincount(ei_np.ravel(), minlength=N_prime)
            core_mask = deg > 0
            old_to_new = np.full(N_prime, -1, dtype=np.int64)
            n_new = 0
            for old_i in range(N_prime):
                if core_mask[old_i]:
                    old_to_new[old_i] = n_new
                    n_new += 1
            N_eff = n_new
            if N_eff < 2:
                continue
            src_e = old_to_new[ei_np[0]]
            dst_e = old_to_new[ei_np[1]]
            edge_index_eff = torch.from_numpy(np.stack([src_e, dst_e])).long()

            # 3. Representative local ids (full 0..N'-1), then map to core 0..N_eff-1 (-1 if isolated)
            rep_local_full = []
            for p in range(src_num_patches):
                gid = int(src_padded_nodes_neighbor_ids[b, p * self.patch_size])
                rep_local_full.append(id_to_local[gid] if (gid > 0 and gid in id_to_local) else -1)
            for p in range(dst_num_patches):
                gid = int(dst_padded_nodes_neighbor_ids[b, p * self.patch_size])
                rep_local_full.append(id_to_local[gid] if (gid > 0 and gid in id_to_local) else -1)
            rep_local = []
            for old in rep_local_full:
                if old < 0:
                    rep_local.append(-1)
                else:
                    nn_idx = int(old_to_new[old])
                    rep_local.append(nn_idx if nn_idx >= 0 else -1)

            # 4. Compute RRWP matrix M: (K, N_eff, N_eff)
            try:
                ei = edge_index_eff.to(dev)
                A = to_dense_adj(ei, max_num_nodes=N_eff).squeeze(0).float()  # (N_eff, N_eff)

                # Row-normalised transition matrix T = D^{-1} A
                row_deg = A.sum(dim=1)                      # (N_eff,)
                deg_inv = torch.zeros_like(row_deg)
                deg_inv[row_deg > 0] = 1.0 / row_deg[row_deg > 0]
                T = deg_inv.unsqueeze(1) * A                # (N_eff, N_eff)

                # Stack walk powers: [I, T, T^2, ..., T^{K-1}] (add_identity=True)
                # or                  [T, T^2, ..., T^{K}]      (add_identity=False)
                pe_list = []
                if self.rrwp_add_identity:
                    pe_list.append(torch.eye(N_eff, dtype=torch.float, device=dev))
                pe_list.append(T)
                out = T
                # fill remaining slots up to K total items
                while len(pe_list) < K:
                    out = out @ T
                    pe_list.append(out)

                M = torch.stack(pe_list, dim=0)  # (K, N_eff, N_eff), values in [0, 1]
            except Exception:
                continue

            # 5. Vectorised fill of c_rrwp[b]
            valid = [(i, rep_local[i]) for i in range(L) if rep_local[i] >= 0]
            if len(valid) < 2:
                continue
            patch_idx = torch.tensor([x[0] for x in valid], dtype=torch.long, device=dev)
            local_idx = torch.tensor([x[1] for x in valid], dtype=torch.long, device=dev)
            # sub_M: (K, len_valid, len_valid) -> permute -> (len_valid, len_valid, K)
            sub_M = M[:, local_idx][:, :, local_idx].permute(1, 2, 0)
            c_rrwp[b, patch_idx[:, None], patch_idx[None, :], :] = sub_M

        # Linear(K -> 1) -> (B, L, L, 1) -> (B, L, L)
        bias = self.rrwp_linear(c_rrwp).squeeze(-1)
        return bias

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
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
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            src_padded_nodes_appearances.append(torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            dst_padded_nodes_appearances.append(torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
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

    def forward(self, inputs: torch.Tensor, struct_bias: torch.Tensor = None):
        """
        :param inputs: Tensor, shape (batch_size, num_patches, attention_dim)
        :param struct_bias: optional Tensor, shape (batch_size, num_patches, num_patches).
               Additive bias on attention logits (RRWP structural bias).
        """
        B, L, _ = inputs.shape
        transposed_inputs = inputs.transpose(0, 1)
        transposed_inputs = self.norm_layers[0](transposed_inputs)

        attn_mask = None
        if struct_bias is not None:
            # (B, L, L) -> (B, 1, L, L) -> (B, H, L, L) -> (B*H, L, L)
            attn_mask = struct_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, L, L)

        hidden_states = self.multi_head_attention(
            query=transposed_inputs, key=transposed_inputs, value=transposed_inputs,
            attn_mask=attn_mask
        )[0].transpose(0, 1)
        outputs = inputs + self.dropout(hidden_states)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        outputs = outputs + self.dropout(hidden_states)
        return outputs
