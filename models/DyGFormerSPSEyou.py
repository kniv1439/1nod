import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.nn import MultiheadAttention
from scipy.sparse import csr_matrix

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

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


if _NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _spse_build_batch_numba(
        all_padded_ids,    # (B, L*patch_size)  int64 — src patches first, then dst patches
        interact_times,    # (B,)               float64
        L,                 # int64 — total number of patches (src_P + dst_P)
        patch_size,        # int64
        ns_flat_ids,       # CSR flat neighbor ids    int64
        ns_flat_times,     # CSR flat neighbor times  float64
        ns_csr_offsets,    # (n_total+1,)             int64
        ns_deg,            # (n_total,)               int64
        n_total,           # int64
        max_sub_nodes,     # int64
        out,               # (B, L, L, 2)             float32
    ):
        """
        All-nodes Numba JIT + prange kernel for SPSE walk-count bias.

        For each batch item b:
          1. Collect all non-zero node IDs from ALL L patches → all_ids.
          2. Build the temporal subgraph (edges with timestamp < t).
          3. Compute A (adjacency, N_eff×N_eff) and A² = A@A.
          4. For each patch pair (pi, pj), sum A[u,v] and A²[u,v] over
             u ∈ patch_pi and v ∈ patch_pj → out[b, pi, pj, 0/1].
        Each batch item writes to its own rows — no write conflicts in prange.
        """
        B = all_padded_ids.shape[0]

        for b in prange(B):
            t = interact_times[b]

            # ── 1. Collect all non-zero nodes from all patches ────────────
            n_all_raw = 0
            for p in range(L):
                for k in range(patch_size):
                    if all_padded_ids[b, p * patch_size + k] > 0:
                        n_all_raw += 1

            if n_all_raw == 0:
                continue

            all_raw = np.empty(n_all_raw, dtype=np.int64)
            j = 0
            for p in range(L):
                for k in range(patch_size):
                    v = all_padded_ids[b, p * patch_size + k]
                    if v > 0:
                        all_raw[j] = v
                        j += 1

            all_ids = np.unique(all_raw)
            N_prime = len(all_ids)
            if N_prime < 2:
                continue
            if max_sub_nodes > 0 and N_prime > max_sub_nodes:
                continue

            # ── 2. Valid nodes with temporal neighbors ────────────────────
            valid_us = np.empty(N_prime, dtype=np.int64)
            n_valid_us = 0
            for uid in all_ids:
                if uid < n_total and ns_deg[uid] > 0:
                    valid_us[n_valid_us] = uid
                    n_valid_us += 1
            if n_valid_us == 0:
                continue

            # ── 3. Build edge list (undirected, within all_ids, time < t) ─
            max_e = n_valid_us * N_prime + 1
            edge_a_buf = np.empty(max_e, dtype=np.int64)
            edge_c_buf = np.empty(max_e, dtype=np.int64)
            n_edges = 0

            for ui in range(n_valid_us):
                u = valid_us[ui]
                u_local = np.searchsorted(all_ids, u)

                s = ns_csr_offsets[u]
                e_end = ns_csr_offsets[u + 1]
                if e_end <= s:
                    continue

                cutoff = np.searchsorted(ns_flat_times[s:e_end], t)
                if cutoff <= 0:
                    continue

                for k in range(cutoff):
                    v = ns_flat_ids[s + k]
                    if v == u:
                        continue
                    v_idx = np.searchsorted(all_ids, v)
                    if v_idx >= N_prime or all_ids[v_idx] != v:
                        continue
                    if u_local < v_idx:
                        a_val = u_local
                        c_val = v_idx
                    else:
                        a_val = v_idx
                        c_val = u_local
                    if n_edges < max_e:
                        edge_a_buf[n_edges] = a_val
                        edge_c_buf[n_edges] = c_val
                        n_edges += 1

            if n_edges == 0:
                continue

            # ── 4. Deduplicate edges ──────────────────────────────────────
            enc = edge_a_buf[:n_edges] * (N_prime + 1) + edge_c_buf[:n_edges]
            enc_sorted = np.sort(enc)
            n_uniq = 1
            for i in range(1, n_edges):
                if enc_sorted[i] != enc_sorted[i - 1]:
                    n_uniq += 1
            da = np.empty(n_uniq, dtype=np.int64)
            dc = np.empty(n_uniq, dtype=np.int64)
            da[0] = enc_sorted[0] // (N_prime + 1)
            dc[0] = enc_sorted[0] %  (N_prime + 1)
            jj = 1
            for i in range(1, n_edges):
                if enc_sorted[i] != enc_sorted[i - 1]:
                    da[jj] = enc_sorted[i] // (N_prime + 1)
                    dc[jj] = enc_sorted[i] %  (N_prime + 1)
                    jj += 1

            # ── 5. Compact to N_eff connected nodes ───────────────────────
            deg_arr = np.zeros(N_prime, dtype=np.int64)
            for i in range(n_uniq):
                deg_arr[da[i]] += 1
                deg_arr[dc[i]] += 1
            n_eff = 0
            for i in range(N_prime):
                if deg_arr[i] > 0:
                    n_eff += 1
            if n_eff < 2:
                continue
            old_to_new = np.full(N_prime, -1, dtype=np.int64)
            ni = 0
            for i in range(N_prime):
                if deg_arr[i] > 0:
                    old_to_new[i] = ni
                    ni += 1
            src_e_arr = np.empty(n_uniq, dtype=np.int64)
            dst_e_arr = np.empty(n_uniq, dtype=np.int64)
            for i in range(n_uniq):
                src_e_arr[i] = old_to_new[da[i]]
                dst_e_arr[i] = old_to_new[dc[i]]

            # ── 6. Build A_full (n_eff × n_eff) adjacency ────────────────
            A_full = np.zeros((n_eff, n_eff), dtype=np.float32)
            for i in range(n_uniq):
                A_full[src_e_arr[i], dst_e_arr[i]] = 1.0
                A_full[dst_e_arr[i], src_e_arr[i]] = 1.0

            # ── 7. A² = A @ A (2-hop walk counts) ────────────────────────
            A2_full = A_full @ A_full

            # ── 8. Aggregate per patch pair via indicator sum ─────────────
            for pi in range(L):
                for pj in range(L):
                    sum_A  = np.float32(0.0)
                    sum_A2 = np.float32(0.0)

                    for ki in range(patch_size):
                        gid_i = all_padded_ids[b, pi * patch_size + ki]
                        if gid_i <= 0:
                            continue
                        idx_i = np.searchsorted(all_ids, gid_i)
                        if idx_i >= N_prime or all_ids[idx_i] != gid_i:
                            continue
                        eff_i = old_to_new[idx_i]
                        if eff_i < 0:
                            continue

                        for kj in range(patch_size):
                            gid_j = all_padded_ids[b, pj * patch_size + kj]
                            if gid_j <= 0:
                                continue
                            idx_j = np.searchsorted(all_ids, gid_j)
                            if idx_j >= N_prime or all_ids[idx_j] != gid_j:
                                continue
                            eff_j = old_to_new[idx_j]
                            if eff_j < 0:
                                continue

                            sum_A  += A_full[eff_i, eff_j]
                            sum_A2 += A2_full[eff_i, eff_j]

                    out[b, pi, pj, 0] = sum_A
                    out[b, pi, pj, 1] = sum_A2
else:
    def _spse_build_batch_numba(*args, **kwargs):
        raise RuntimeError("Numba not available")


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
                 spse_max_subgraph_nodes: int = 200,
                 spse_precomputed_dir: str = None,
                 spse_lazy_load: bool = False):
        """
        DyGFormer model with SPSE structural bias on patch-patch attention.

        子图节点范围：所有 patch 内的全部非零节点（与 precompute_dynamic_spse.py 一致）。
        若提供 spse_precomputed_dir，则在训练/评估时直接从预计算文件查找 SPSE bias，
        无需在线计算（正样本）；负样本 edge_ids=None 时 bias 为 0。
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

        # ── 预计算 SPSE 数据（可选）────────────────────────────────────────
        # spse_lazy_load=False（默认）：全量加载所有 chunk_*.pkl 合并为一个大字典（适合小数据集）。
        # spse_lazy_load=True（UNvote 等大数据集专用）：启动时只读 meta.json，
        #   训练中按需按 chunk 加载，LRU 保留最近 _SPSE_MAX_CACHED_CHUNKS 个 chunk，
        #   内存占用从「百万条全表」降至「常数个 chunk」。
        _SPSE_MAX_CACHED_CHUNKS = 8
        self._spse_precomputed = None        # 全量 dict（非懒加载时使用）
        self._spse_precomp_dir = None        # 懒加载：预计算目录路径
        self._spse_chunk_size  = None        # 懒加载：单个 chunk 的交互数
        self._spse_chunk_cache: dict = {}    # 懒加载：已加载的 chunk {chunk_id: dict}
        self._spse_chunk_lru:   list = []    # 懒加载：LRU 顺序
        self._spse_max_cached_chunks = _SPSE_MAX_CACHED_CHUNKS

        if spse_precomputed_dir is not None:
            import json as _json
            # 先读 meta.json（两种模式都需要）
            _meta_path = os.path.join(spse_precomputed_dir, 'meta.json')
            _dist_len = spse_dist_len
            _chunk_size = 5000  # 默认值，meta.json 里会覆盖
            if os.path.exists(_meta_path):
                with open(_meta_path) as _f:
                    _meta = _json.load(_f)
                _dist_len   = int(_meta.get('dist_len', spse_dist_len))
                _chunk_size = int(_meta.get('chunk_size', 5000))
            if _dist_len != self.spse_bias_feat_dim:
                self.spse_bias_feat_dim = _dist_len
                self.spse_linear = nn.Linear(self.spse_bias_feat_dim, 1, bias=True)

            if spse_lazy_load:
                # ── 懒加载模式（UNvote 专用）：仅记元数据，不读任何 pkl ────
                self._spse_precomp_dir = spse_precomputed_dir
                self._spse_chunk_size  = _chunk_size
                print(f'[DyGFormerSPSE] Lazy-load mode (UNvote): '
                      f'chunk_size={_chunk_size}, max_cached_chunks={_SPSE_MAX_CACHED_CHUNKS}, '
                      f'dist_len={self.spse_bias_feat_dim}')
            else:
                # ── 全量加载模式（默认，适合小/中型数据集）──────────────────
                import glob as _glob
                import pickle as _pickle
                precomp: dict = {}
                for _path in sorted(_glob.glob(os.path.join(spse_precomputed_dir, 'chunk_*.pkl'))):
                    with open(_path, 'rb') as _f:
                        precomp.update(_pickle.load(_f))
                self._spse_precomputed = precomp
                print(f'[DyGFormerSPSE] Loaded {len(precomp)} precomputed SPSE entries '
                      f'from {spse_precomputed_dir} (dist_len={self.spse_bias_feat_dim})')

        n_workers = os.cpu_count() or 4
        self._executor = ThreadPoolExecutor(max_workers=n_workers)

        self._timers = {
            'neighbor_sampling': 0.0,
            'pad_sequences':     0.0,
            'co_occurrence':     0.0,
            'get_features':      0.0,
            'get_patches':       0.0,
            'projection':        0.0,
            'spse_bias':         0.0,
            'transformer':       0.0,
            'output':            0.0,
        }

        # SPSE 结果缓存：key = (padded_b.tobytes(), t)，value = M_sub (L, L, K) float32 数组
        # 命中则直接复用，避免跨 epoch 重复计算相同子图。
        # 上限 20000 条，超出后停止写入（不驱逐，保守策略）。
        self._spse_cache: dict = {}
        self._spse_cache_max: int = 20000

    def reset_timers(self):
        for k in self._timers:
            self._timers[k] = 0.0

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

        # CSR 扁平格式：供 Numba kernel 使用（连续内存，避免 Python list of arrays）
        sizes = self._ns_deg_np
        self._ns_csr_offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        self._ns_csr_offsets[1:] = np.cumsum(sizes)
        total = int(self._ns_csr_offsets[-1])
        if total > 0:
            self._ns_flat_ids   = np.concatenate([x.astype(np.int64)   for x in self._ns_ids_np])
            self._ns_flat_times = np.concatenate([x.astype(np.float64) for x in self._ns_times_np])
        else:
            self._ns_flat_ids   = np.array([], dtype=np.int64)
            self._ns_flat_times = np.array([], dtype=np.float64)

    def _load_spse_module(self):
        """Import count_self_avoiding_paths once to avoid thread-unsafe sys.path mutation."""
        import sys
        _spse_dir = os.path.join(os.path.dirname(__file__), '..', 'Graph-SPSE-Encoding')
        if _spse_dir not in sys.path:
            sys.path.insert(0, _spse_dir)
        from count_self_avoiding_paths import get_simple_paths_count
        self._get_simple_paths_count = get_simple_paths_count

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                  node_interact_times: np.ndarray, edge_ids: np.ndarray = None):
        """
        compute source and destination node temporal embeddings.
        edge_ids: 可选，正样本的全局边 id（来自 CSV 的 idx 列）。
                  提供时用于从预计算 SPSE 字典中查找 bias；不提供则在线计算（或返回 0）。
        """
        _t = time.perf_counter

        # ── 1. 邻居采样 ──────────────────────────────────────────────────
        t0 = _t()
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, node_interact_times=node_interact_times
            )

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids, node_interact_times=node_interact_times
            )
        self._timers['neighbor_sampling'] += _t() - t0

        # ── 2. 序列填充 ───────────────────────────────────────────────────
        t0 = _t()
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
        self._timers['pad_sequences'] += _t() - t0

        # ── 3. 共现特征编码 ───────────────────────────────────────────────
        t0 = _t()
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids
            )
        self._timers['co_occurrence'] += _t() - t0

        # ── 4. 节点/边/时间特征提取 ───────────────────────────────────────
        t0 = _t()
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
        self._timers['get_features'] += _t() - t0

        # ── 5. Patch 构建 ─────────────────────────────────────────────────
        t0 = _t()
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
        self._timers['get_patches'] += _t() - t0

        # ── 6. 投影层 ─────────────────────────────────────────────────────
        t0 = _t()
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)
        self._timers['projection'] += _t() - t0

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

        # ── 7. SPSE bias 计算 ─────────────────────────────────────────────
        t0 = _t()
        struct_bias = self.compute_spse_bias(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            node_interact_times=node_interact_times,
            src_num_patches=src_num_patches,
            dst_num_patches=dst_num_patches,
            edge_ids=edge_ids,
        )
        self._timers['spse_bias'] += _t() - t0

        # ── 8. Transformer 前向 ───────────────────────────────────────────
        t0 = _t()
        for transformer in self.transformers:
            patches_data = transformer(patches_data, struct_bias=struct_bias)
        self._timers['transformer'] += _t() - t0

        src_patches_data = patches_data[:, : src_num_patches, :]
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # ── 9. 输出层 ─────────────────────────────────────────────────────
        t0 = _t()
        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)
        self._timers['output'] += _t() - t0

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

    def _get_precomp_entry(self, eid: int):
        """从全量 dict（小数据集）或懒加载 chunk（UNvote 等大数据集）中取单条 entry。"""
        # 全量模式（非 UNvote 的默认路径）
        if self._spse_precomputed is not None:
            return self._spse_precomputed.get(eid)

        # 懒加载模式（UNvote 专用）
        if self._spse_precomp_dir is None:
            return None

        chunk_id = eid // self._spse_chunk_size
        cache = self._spse_chunk_cache

        if chunk_id not in cache:
            # LRU 驱逐：超出上限时移除最旧的 chunk
            if len(cache) >= self._spse_max_cached_chunks:
                evict_id = self._spse_chunk_lru.pop(0)
                del cache[evict_id]
            # 加载对应 chunk 文件
            import pickle as _pickle
            chunk_path = os.path.join(self._spse_precomp_dir,
                                      f'chunk_{chunk_id:05d}.pkl')
            if os.path.exists(chunk_path):
                with open(chunk_path, 'rb') as _f:
                    cache[chunk_id] = _pickle.load(_f)
            else:
                cache[chunk_id] = {}
            self._spse_chunk_lru.append(chunk_id)
        else:
            # 命中缓存 → 更新 LRU 顺序（移至末尾）
            self._spse_chunk_lru.remove(chunk_id)
            self._spse_chunk_lru.append(chunk_id)

        return cache[chunk_id].get(eid)

    def compute_spse_bias(self,
                          src_padded_nodes_neighbor_ids: np.ndarray,
                          dst_padded_nodes_neighbor_ids: np.ndarray,
                          node_interact_times: np.ndarray,
                          src_num_patches: int,
                          dst_num_patches: int,
                          edge_ids: np.ndarray = None) -> torch.Tensor:
        """
        Compute the SPSE-based structural bias matrix for patch-patch attention.

        若预计算模式（全量或懒加载）激活且提供了 edge_ids，则直接查表：
            - vals 已是 log1p(count)，直接填入，不再调用 torch.log1p
            - 找不到的 entry（负样本、预计算缺失）bias 保持 0
        否则走在线计算路径（Numba 或 Python 回退），计算后调用 torch.log1p。
        """
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()

        dev = self.device
        B = len(node_interact_times)
        L = src_num_patches + dst_num_patches
        K = self.spse_bias_feat_dim

        # 预计算模式激活标志（全量 dict 或懒加载目录，二者选其一）
        _precomp_active = (self._spse_precomputed is not None
                           or self._spse_precomp_dir is not None)

        # ── 负样本快速路径：预计算模式下 edge_ids=None 直接返回零 bias ────
        if _precomp_active and edge_ids is None:
            return torch.zeros(B, L, L, device=dev)

        c_spse_cpu = np.zeros((B, L, L, K), dtype=np.float32)

        # ── 预计算查找路径 ─────────────────────────────────────────────────
        if _precomp_active and edge_ids is not None:
            _eids = edge_ids.cpu().numpy() if isinstance(edge_ids, torch.Tensor) else np.asarray(edge_ids)
            for b, eid in enumerate(_eids):
                entry = self._get_precomp_entry(int(eid))
                if entry is None:
                    continue
                src_P_pre = int(entry['src_P'])
                rows = entry['row'].astype(np.int32)
                cols = entry['col'].astype(np.int32)
                vals = entry['vals'].astype(np.float32)   # (nnz, dist_len), 已 log1p
                # 将预计算中基于 (src_P_pre, dst_P_pre) 的 patch 索引
                # 映射到当前 batch 的 (src_num_patches, dst_num_patches)
                batch_rows = np.where(rows < src_P_pre, rows,
                                      src_num_patches + (rows - src_P_pre))
                batch_cols = np.where(cols < src_P_pre, cols,
                                      src_num_patches + (cols - src_P_pre))
                ok = (batch_rows < L) & (batch_cols < L)
                c_spse_cpu[b, batch_rows[ok], batch_cols[ok], :] = vals[ok]
            # vals 已是 log1p，直接过 linear 层
            c_spse = torch.from_numpy(c_spse_cpu).to(dev)
            bias = self.spse_linear(c_spse).squeeze(-1)
            return bias

        n_total = len(self._ns_ids_np)
        patch_size = self.patch_size
        max_sub_nodes = self.spse_max_subgraph_nodes
        use_walk_counts = self.spse_use_adjacency_walk_counts

        # Concatenate src and dst padded sequences: (B, L * patch_size)
        # patch p corresponds to columns [p*patch_size : (p+1)*patch_size]
        all_padded_combined = np.concatenate([
            src_padded_nodes_neighbor_ids.astype(np.int64),
            dst_padded_nodes_neighbor_ids.astype(np.int64),
        ], axis=1)

        if use_walk_counts and _NUMBA_AVAILABLE:
            # ── Numba 快速路径：真多核并行，无 GIL ────────────────────────
            _spse_build_batch_numba(
                all_padded_combined,
                node_interact_times.astype(np.float64),
                np.int64(L),
                np.int64(patch_size),
                self._ns_flat_ids,
                self._ns_flat_times,
                self._ns_csr_offsets,
                self._ns_deg_np,
                np.int64(n_total),
                np.int64(max_sub_nodes),
                c_spse_cpu,
            )
        else:
            # ── Python 回退路径（Numba 不可用 或 use_walk_counts=False）────
            ns_ids = self._ns_ids_np
            ns_times = self._ns_times_np
            ns_deg = self._ns_deg_np

            spse_min_root = self.spse_min_root_nodes
            spse_max_root = self.spse_max_root_nodes
            spse_root_fact = self.spse_root_nodes_fact
            spse_dist_len = self.spse_dist_len
            spse_bfs_depth = self.spse_max_bfs_depth
            spse_tries = self.spse_max_tries_per_depth
            get_spc = self._get_simple_paths_count

            spse_cache = self._spse_cache
            spse_cache_max = self._spse_cache_max

            def _process_sample(b: int):
                t = float(node_interact_times[b])
                padded_b = all_padded_combined[b]  # (L * patch_size,)

                # ── Collect ALL non-zero nodes per patch ──────────────────
                all_patch_nodes_list = []
                for p in range(L):
                    block = padded_b[p * patch_size: (p + 1) * patch_size]
                    nodes_p = block[block > 0]
                    all_patch_nodes_list.append(nodes_p)

                all_node_parts = [p for p in all_patch_nodes_list if len(p) > 0]
                if not all_node_parts:
                    return

                all_ids = np.unique(np.concatenate(all_node_parts))
                N_prime = len(all_ids)
                if N_prime < 2:
                    return
                if max_sub_nodes > 0 and N_prime > max_sub_nodes:
                    return

                cache_key = (padded_b.tobytes(), t)
                if cache_key in spse_cache:
                    cached = spse_cache[cache_key]
                    if cached is not None:
                        c_spse_cpu[b, :, :, :] = cached
                    return

                # ── Build edge list ────────────────────────────────────────
                valid_us = all_ids[all_ids < n_total]
                has_nbr = ns_deg[valid_us] > 0
                valid_us_nbr = valid_us[has_nbr]

                if len(valid_us_nbr) == 0:
                    spse_cache[cache_key] = None
                    return

                u_local_arr = np.searchsorted(all_ids, valid_us_nbr).astype(np.int64)
                u_local_dict = {int(u): int(loc) for u, loc in zip(valid_us_nbr, u_local_arr)}

                edge_a_list = []
                edge_c_list = []
                for u in valid_us_nbr:
                    u = int(u)
                    nbr_ids_u = ns_ids[u]
                    nbr_times_u = ns_times[u]
                    if len(nbr_ids_u) == 0:
                        continue
                    cutoff = int(np.searchsorted(nbr_times_u, t, side="left"))
                    if cutoff <= 0:
                        continue
                    hist_nbr_ids = nbr_ids_u[:cutoff]
                    idx = np.searchsorted(all_ids, hist_nbr_ids)
                    idx_clip = np.clip(idx, 0, N_prime - 1)
                    in_mask = (hist_nbr_ids != u) & (all_ids[idx_clip] == hist_nbr_ids)
                    if not np.any(in_mask):
                        continue
                    v_locals = idx_clip[in_mask]
                    u_local = u_local_dict[u]
                    u_locs = np.full(len(v_locals), u_local, dtype=np.int64)
                    edge_a_list.append(np.minimum(u_locs, v_locals))
                    edge_c_list.append(np.maximum(u_locs, v_locals))

                if not edge_a_list:
                    spse_cache[cache_key] = None
                    return

                all_a = np.concatenate(edge_a_list)
                all_c = np.concatenate(edge_c_list)
                edge_encoded = all_a * (N_prime + 1) + all_c
                _, unique_idx = np.unique(edge_encoded, return_index=True)
                dedup_src = all_a[unique_idx]
                dedup_dst = all_c[unique_idx]

                # ── Compact to N_eff connected nodes ───────────────────────
                ei_np = np.stack([dedup_src, dedup_dst], axis=0)
                deg = np.bincount(ei_np.ravel(), minlength=N_prime)
                core_indices = np.where(deg > 0)[0]
                N_eff = len(core_indices)
                if N_eff < 2:
                    spse_cache[cache_key] = None
                    return

                old_to_new = np.full(N_prime, -1, dtype=np.int64)
                old_to_new[core_indices] = np.arange(N_eff, dtype=np.int64)

                src_e = old_to_new[ei_np[0]]
                dst_e = old_to_new[ei_np[1]]

                try:
                    # ── Build indicator (L × N_eff) ────────────────────────
                    indicator = np.zeros((L, N_eff), dtype=np.float32)
                    for pi, patch_nodes in enumerate(all_patch_nodes_list):
                        if len(patch_nodes) == 0:
                            continue
                        idx = np.searchsorted(all_ids, patch_nodes)
                        idx_clip = np.clip(idx, 0, N_prime - 1)
                        in_mask = all_ids[idx_clip] == patch_nodes
                        local_idx = idx_clip[in_mask]
                        eff_idx = old_to_new[local_idx]
                        eff_valid = eff_idx[eff_idx >= 0]
                        if len(eff_valid) > 0:
                            indicator[pi, eff_valid] = 1.0

                    if use_walk_counts:
                        # A (N_eff × N_eff) adjacency matrix
                        A_full = np.zeros((N_eff, N_eff), dtype=np.float32)
                        A_full[src_e, dst_e] = 1.0
                        A_full[dst_e, src_e] = 1.0
                        A2_full = (A_full @ A_full).astype(np.float32)

                        # Patch-pair aggregation: out[pi,pj,k] = Σ_{u∈pi,v∈pj} M[u,v,k]
                        agg_A  = indicator @ A_full  @ indicator.T   # (L, L)
                        agg_A2 = indicator @ A2_full @ indicator.T   # (L, L)
                        M_sub = np.stack([agg_A, agg_A2], axis=-1)   # (L, L, 2)

                    else:
                        # Full SPSE via get_simple_paths_count
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
                        M3d = M.squeeze(0).float()   # (dist_len, N_eff, N_eff)
                        indicator_t = torch.from_numpy(indicator)
                        # row_agg[pi,k,v] = Σ_{u∈pi} M[k,u,v]  → (L, dist_len, N_eff)
                        row_agg = torch.einsum('lu,kuv->lkv', indicator_t, M3d)
                        # patch_agg[pi,pj,k] = Σ_{v∈pj} row_agg[pi,k,v]  → (L, L, dist_len)
                        patch_agg = torch.einsum('lkv,mv->lmk', row_agg, indicator_t)
                        M_sub = patch_agg.numpy().astype(np.float32)   # (L, L, dist_len)

                except Exception:
                    spse_cache[cache_key] = None
                    return

                c_spse_cpu[b, :, :, :] = M_sub
                if len(spse_cache) < spse_cache_max:
                    spse_cache[cache_key] = M_sub.copy()

            if B > 1:
                list(self._executor.map(_process_sample, range(B)))
            else:
                _process_sample(0)

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
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
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