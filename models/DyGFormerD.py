import os
import glob
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from scipy import sparse as sp
from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class DyGFormerD(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu',
                 use_rwpe: bool = False, rwpe_dim: int = 8, use_lpu: bool = None, use_rpe: bool = None,
                 rwpe_precomputed_dir: str = None, rwpe_lazy_load: bool = False, max_subgraph_size: int = 512):
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
        :param use_rwpe: bool, 是否使用 RWPE 位置编码（如果提供了 use_rpe，则优先使用 use_rpe）
        :param rwpe_dim: int, RWPE 的维度
        :param use_lpu: bool, 兼容性参数（未使用）
        :param use_rpe: bool, 是否使用位置编码（会映射到 use_rwpe）
        :param rwpe_precomputed_dir: str, 预计算 RWPE 目录。
            key_type=node_t（推荐，rwpe_node_t_*）：正负样本均按 (node_id, t) 查表，miss 时同算子在线算并懒缓存。
            key_type=edge_id（旧版 rwpe_precomputed_*）：仅正样本按 edge_id 查表。
        :param rwpe_lazy_load: bool, 大数据集懒加载 chunk
        :param max_subgraph_size: int, RWPE 子图 unique 节点数上限（与预计算一致）
        """
        super(DyGFormerD, self).__init__()

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
        self.max_subgraph_size = max_subgraph_size

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        # 如果提供了 use_rpe，则优先使用它；否则使用 use_rwpe
        self.use_rwpe = use_rpe if use_rpe is not None else use_rwpe
        self.rwpe_dim = rwpe_dim
        # use_lpu 参数保留用于兼容性，但当前未使用
        
        # 如果使用 RWPE，增加一个通道
        self.num_channels = 5 if self.use_rwpe else 4

        # 与原版一致：先拼 raw patch，再投影到 channel_embedding_dim
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        if self.use_rwpe:
            self.projection_layer['rwpe'] = nn.Linear(
                in_features=self.patch_size * self.rwpe_dim,
                out_features=self.channel_embedding_dim,
                bias=True
            )

        # ── 预计算 RWPE（可选）──────────────────────────────────────────
        self._rwpe_key_type = None  # 'node_t' | 'edge_id'
        self._rwpe_precomputed = None
        self._rwpe_precomp_dir = None
        self._rwpe_chunk_size = None
        self._rwpe_chunk_cache = {}
        self._rwpe_chunk_lru = []
        self._rwpe_max_cached_chunks = 8
        self._rwpe_eid_to_row = None  # edge_id mode: edge_id -> row
        self._rwpe_key_to_row = None  # node_t mode: (node, t) -> row
        self._rwpe_runtime_cache = {}  # node_t misses / lazy_cache merges
        self._rwpe_runtime_dirty = {}  # newly computed since last flush
        self._rwpe_lazy_cache_dir = None
        self._rwpe_runtime_new_count = 0

        if self.use_rwpe and rwpe_precomputed_dir is not None:
            self._init_rwpe_precomputed(rwpe_precomputed_dir, rwpe_lazy_load)

        # Transformer 输入维度固定为 num_channels * channel_embedding_dim（与原版一致）
        original_attention_dim = self.num_channels * self.channel_embedding_dim
        if original_attention_dim % self.num_heads != 0:
            self.attention_dim = ((original_attention_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
            self.attention_dim_adjust = nn.Linear(in_features=original_attention_dim, out_features=self.attention_dim, bias=True)
            self.attention_dim_revert = nn.Linear(in_features=self.attention_dim, out_features=original_attention_dim, bias=True)
        else:
            self.attention_dim = original_attention_dim
            self.attention_dim_adjust = None
            self.attention_dim_revert = None

        self.original_attention_dim = original_attention_dim

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.attention_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=original_attention_dim, out_features=self.node_feat_dim, bias=True)

    @staticmethod
    def _rwpe_node_t_key(node_id, t):
        return (int(node_id), float(t))

    def _init_rwpe_precomputed(self, rwpe_precomputed_dir: str, rwpe_lazy_load: bool):
        """Load precomputed RWPE metadata / chunks (node_t or legacy edge_id)."""
        meta_path = os.path.join(rwpe_precomputed_dir, 'meta.json')
        chunk_size = 5000
        key_type = 'edge_id'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            key_type = meta.get('key_type', 'edge_id')
            # Infer node_t from directory name if meta omitted
            if key_type == 'edge_id' and 'rwpe_node_t_' in os.path.basename(rwpe_precomputed_dir.rstrip('/')):
                key_type = 'node_t'
            meta_dim = int(meta.get('rwpe_dim', self.rwpe_dim))
            meta_seq = int(meta.get('max_input_sequence_length', self.max_input_sequence_length))
            meta_ms = int(meta.get('max_subgraph_size', self.max_subgraph_size))
            chunk_size = int(meta.get('chunk_size', 5000))
            if meta_dim != self.rwpe_dim:
                raise ValueError(
                    f'RWPE precompute rwpe_dim={meta_dim} != model rwpe_dim={self.rwpe_dim}'
                )
            if meta_seq != self.max_input_sequence_length:
                raise ValueError(
                    f'RWPE precompute max_input_sequence_length={meta_seq} != '
                    f'model={self.max_input_sequence_length}'
                )
            self.max_subgraph_size = meta_ms
        elif 'rwpe_node_t_' in os.path.basename(rwpe_precomputed_dir.rstrip('/')):
            key_type = 'node_t'

        self._rwpe_key_type = key_type
        self._rwpe_lazy_cache_dir = os.path.join(rwpe_precomputed_dir, 'lazy_cache')

        # Merge any previously flushed lazy_cache files into runtime cache
        if key_type == 'node_t' and os.path.isdir(self._rwpe_lazy_cache_dir):
            for path in sorted(glob.glob(os.path.join(self._rwpe_lazy_cache_dir, '*.pkl'))):
                try:
                    with open(path, 'rb') as f:
                        self._rwpe_runtime_cache.update(pickle.load(f))
                except Exception:
                    pass
            if self._rwpe_runtime_cache:
                print(
                    f'[DyGFormerD] Loaded {len(self._rwpe_runtime_cache)} lazy_cache RWPE entries'
                )

        if rwpe_lazy_load:
            self._rwpe_precomp_dir = rwpe_precomputed_dir
            self._rwpe_chunk_size = chunk_size
            if key_type == 'node_t':
                keys_path = os.path.join(rwpe_precomputed_dir, 'keys.npz')
                if os.path.exists(keys_path):
                    kz = np.load(keys_path)
                    node_ids = kz['node_ids']
                    times = kz['times']
                    self._rwpe_key_to_row = {
                        (int(n), float(t)): i
                        for i, (n, t) in enumerate(zip(node_ids.tolist(), times.tolist()))
                    }
                print(
                    f'[DyGFormerD] RWPE node_t lazy-load: dir={rwpe_precomputed_dir}, '
                    f'chunk_size={chunk_size}, keys={0 if self._rwpe_key_to_row is None else len(self._rwpe_key_to_row)}'
                )
            else:
                eid_path = os.path.join(rwpe_precomputed_dir, 'edge_ids.npy')
                if os.path.exists(eid_path):
                    edge_ids_arr = np.load(eid_path)
                    self._rwpe_eid_to_row = {int(e): i for i, e in enumerate(edge_ids_arr.tolist())}
                print(
                    f'[DyGFormerD] RWPE edge_id lazy-load: dir={rwpe_precomputed_dir}, '
                    f'chunk_size={chunk_size}, rwpe_dim={self.rwpe_dim}'
                )
        else:
            precomp = {}
            for path in sorted(glob.glob(os.path.join(rwpe_precomputed_dir, 'chunk_*.pkl'))):
                with open(path, 'rb') as f:
                    precomp.update(pickle.load(f))
            self._rwpe_precomputed = precomp
            print(
                f'[DyGFormerD] Loaded {len(precomp)} precomputed RWPE entries '
                f'from {rwpe_precomputed_dir} (key_type={key_type}, rwpe_dim={self.rwpe_dim})'
            )

    def _load_chunk_dict(self, chunk_id: int):
        cache = self._rwpe_chunk_cache
        if chunk_id in cache:
            self._rwpe_chunk_lru.remove(chunk_id)
            self._rwpe_chunk_lru.append(chunk_id)
            return cache[chunk_id]

        if len(cache) >= self._rwpe_max_cached_chunks:
            evict_id = self._rwpe_chunk_lru.pop(0)
            del cache[evict_id]
        chunk_path = os.path.join(self._rwpe_precomp_dir, f'chunk_{chunk_id:05d}.pkl')
        if os.path.exists(chunk_path):
            with open(chunk_path, 'rb') as f:
                cache[chunk_id] = pickle.load(f)
        else:
            cache[chunk_id] = {}
        self._rwpe_chunk_lru.append(chunk_id)
        return cache[chunk_id]

    def _get_rwpe_entry_edge(self, eid: int):
        """Lookup legacy precomputed RWPE entry by edge_id."""
        if self._rwpe_precomputed is not None:
            return self._rwpe_precomputed.get(eid)

        if self._rwpe_precomp_dir is None:
            return None

        if self._rwpe_eid_to_row is not None:
            row = self._rwpe_eid_to_row.get(eid)
            if row is None:
                return None
            chunk_id = row // self._rwpe_chunk_size
        else:
            chunk_id = eid // self._rwpe_chunk_size

        return self._load_chunk_dict(chunk_id).get(eid)

    def _get_rwpe_entry_node_t(self, node_id, t):
        """Lookup (node_id, t) RWPE array or None if miss."""
        k = self._rwpe_node_t_key(node_id, t)
        if k in self._rwpe_runtime_cache:
            return self._rwpe_runtime_cache[k]
        if self._rwpe_precomputed is not None:
            return self._rwpe_precomputed.get(k)
        if self._rwpe_precomp_dir is None or self._rwpe_key_to_row is None:
            return None
        row = self._rwpe_key_to_row.get(k)
        if row is None:
            return None
        chunk_id = row // self._rwpe_chunk_size
        return self._load_chunk_dict(chunk_id).get(k)

    def _store_rwpe_runtime(self, node_id, t, pe_array: np.ndarray):
        k = self._rwpe_node_t_key(node_id, t)
        if k in self._rwpe_runtime_cache:
            return
        arr = np.asarray(pe_array, dtype=np.float16)
        self._rwpe_runtime_cache[k] = arr
        self._rwpe_runtime_dirty[k] = arr
        self._rwpe_runtime_new_count += 1
        # Auto-flush every 2000 new misses to disk
        if self._rwpe_runtime_new_count >= 2000:
            self.flush_rwpe_lazy_cache()

    def flush_rwpe_lazy_cache(self):
        """Persist newly computed (node,t) RWPE entries under lazy_cache/."""
        if not self._rwpe_runtime_dirty or self._rwpe_lazy_cache_dir is None:
            return
        os.makedirs(self._rwpe_lazy_cache_dir, exist_ok=True)
        n = len(self._rwpe_runtime_dirty)
        path = os.path.join(
            self._rwpe_lazy_cache_dir,
            f'cache_{n}_{os.getpid()}_{len(self._rwpe_runtime_cache)}.pkl',
        )
        with open(path, 'wb') as f:
            pickle.dump(self._rwpe_runtime_dirty, f, protocol=4)
        self._rwpe_runtime_dirty = {}
        self._rwpe_runtime_new_count = 0
        print(f'[DyGFormerD] Flushed RWPE lazy_cache → {path} ({n} new keys)')

    def _rwpe_precomp_active(self) -> bool:
        return self._rwpe_precomputed is not None or self._rwpe_precomp_dir is not None

    @staticmethod
    def _valid_seq_len(padded_node_ids_1d: np.ndarray) -> int:
        nz = np.flatnonzero(padded_node_ids_1d != 0)
        return int(nz[-1] + 1) if len(nz) > 0 else 0

    def _fill_rwpe_from_array(self, out_row: torch.Tensor, pe: np.ndarray, seq_len: int):
        pe = np.asarray(pe, dtype=np.float32)
        L = min(pe.shape[0], seq_len)
        if L > 0:
            out_row[:L] = torch.from_numpy(pe[:L])

    def _load_rwpe_from_precomp(
        self, edge_ids,
        src_node_ids, dst_node_ids, node_interact_times,
        src_seq_len: int, dst_seq_len: int,
        src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids,
        src_nodes_neighbor_ids_list, dst_nodes_neighbor_ids_list,
    ):
        """
        Legacy edge_id lookup. Load precomputed RWPE and pad/truncate.
        Miss (edge_id not found) falls back to online compute_rwpe_features,
        the same path used for negative samples (fair with node_t mode).
        :return: src_rwpe, dst_rwpe Tensors (B, seq_len, rwpe_dim)
        """
        if isinstance(edge_ids, torch.Tensor):
            edge_ids = edge_ids.detach().cpu().numpy()
        edge_ids = np.asarray(edge_ids)

        B, d = len(edge_ids), self.rwpe_dim
        src_out = torch.zeros(B, src_seq_len, d, device=self.device)
        dst_out = torch.zeros(B, dst_seq_len, d, device=self.device)

        miss = []
        for b, eid in enumerate(edge_ids):
            entry = self._get_rwpe_entry_edge(int(eid))
            if entry is None:
                miss.append(b)
                continue
            src_pe = np.asarray(entry['src'], dtype=np.float32)
            dst_pe = np.asarray(entry['dst'], dtype=np.float32)
            ls = min(src_pe.shape[0], src_seq_len)
            ld = min(dst_pe.shape[0], dst_seq_len)
            if ls > 0:
                src_out[b, :ls] = torch.from_numpy(src_pe[:ls])
            if ld > 0:
                dst_out[b, :ld] = torch.from_numpy(dst_pe[:ld])

        if miss:
            online_src = self.compute_rwpe_features(
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
            )
            online_dst = self.compute_rwpe_features(
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
            )
            for b in miss:
                src_out[b] = online_src[b]
                dst_out[b] = online_dst[b]

        return src_out, dst_out

    def _load_rwpe_node_t(
        self,
        src_node_ids,
        dst_node_ids,
        node_interact_times,
        src_seq_len: int,
        dst_seq_len: int,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        src_nodes_neighbor_ids_list: list,
        dst_nodes_neighbor_ids_list: list,
    ):
        """
        Fair path: both sides lookup (node_id, t); miss → online compute_rwpe_features (same path) + runtime cache.
        """
        if isinstance(src_node_ids, torch.Tensor):
            src_node_ids = src_node_ids.detach().cpu().numpy()
        if isinstance(dst_node_ids, torch.Tensor):
            dst_node_ids = dst_node_ids.detach().cpu().numpy()
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy()

        src_node_ids = np.asarray(src_node_ids)
        dst_node_ids = np.asarray(dst_node_ids)
        node_interact_times = np.asarray(node_interact_times)

        B, d = len(src_node_ids), self.rwpe_dim
        src_out = torch.zeros(B, src_seq_len, d, device=self.device)
        dst_out = torch.zeros(B, dst_seq_len, d, device=self.device)

        miss_src, miss_dst = [], []
        for b in range(B):
            t = float(node_interact_times[b])
            pe_s = self._get_rwpe_entry_node_t(src_node_ids[b], t)
            if pe_s is not None:
                self._fill_rwpe_from_array(src_out[b], pe_s, src_seq_len)
            else:
                miss_src.append(b)
            pe_d = self._get_rwpe_entry_node_t(dst_node_ids[b], t)
            if pe_d is not None:
                self._fill_rwpe_from_array(dst_out[b], pe_d, dst_seq_len)
            else:
                miss_dst.append(b)

        if miss_src:
            online_src = self.compute_rwpe_features(
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
            )
            for b in miss_src:
                src_out[b] = online_src[b]
                L = self._valid_seq_len(src_padded_nodes_neighbor_ids[b])
                if L > 0:
                    pe = online_src[b, :L].detach().cpu().numpy()
                    self._store_rwpe_runtime(src_node_ids[b], node_interact_times[b], pe)

        if miss_dst:
            online_dst = self.compute_rwpe_features(
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
            )
            for b in miss_dst:
                dst_out[b] = online_dst[b]
                L = self._valid_seq_len(dst_padded_nodes_neighbor_ids[b])
                if L > 0:
                    pe = online_dst[b, :L].detach().cpu().numpy()
                    self._store_rwpe_runtime(dst_node_ids[b], node_interact_times[b], pe)

        return src_out, dst_out

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, edge_ids=None):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: optional; only used for legacy edge_id RWPE precompute.
                         node_t mode ignores this (pos/neg both use (node, t)).
        :return:
        """
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

        # 如果使用 RWPE：
        #   node_t 预计算：正负均按 (node,t) 查表，miss 同算子在线+懒缓存
        #   edge_id 旧预计算：仅正样本查表；否则全在线
        if self.use_rwpe:
            src_seq_len = src_padded_nodes_neighbor_ids.shape[1]
            dst_seq_len = dst_padded_nodes_neighbor_ids.shape[1]
            if self._rwpe_precomp_active() and self._rwpe_key_type == 'node_t':
                src_rwpe_features, dst_rwpe_features = self._load_rwpe_node_t(
                    src_node_ids=src_node_ids,
                    dst_node_ids=dst_node_ids,
                    node_interact_times=node_interact_times,
                    src_seq_len=src_seq_len,
                    dst_seq_len=dst_seq_len,
                    src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                    dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                    src_nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                    dst_nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                )
            elif self._rwpe_precomp_active() and self._rwpe_key_type == 'edge_id' and edge_ids is not None:
                src_rwpe_features, dst_rwpe_features = self._load_rwpe_from_precomp(
                    edge_ids=edge_ids,
                    src_node_ids=src_node_ids,
                    dst_node_ids=dst_node_ids,
                    node_interact_times=node_interact_times,
                    src_seq_len=src_seq_len,
                    dst_seq_len=dst_seq_len,
                    src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                    dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                    src_nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                    dst_nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                )
            else:
                src_rwpe_features = self.compute_rwpe_features(
                    padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                    nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                    node_ids=src_node_ids,
                    node_interact_times=node_interact_times
                )
                dst_rwpe_features = self.compute_rwpe_features(
                    padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                    nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                    node_ids=dst_node_ids,
                    node_interact_times=node_interact_times
                )

        # 先拼 raw patch，再投影到 channel_embedding_dim（与原版一致）
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

        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        if self.use_rwpe:
            src_patches_rwpe = self.get_patches_rwpe(rwpe_features=src_rwpe_features, patch_size=self.patch_size)
            dst_patches_rwpe = self.get_patches_rwpe(rwpe_features=dst_rwpe_features, patch_size=self.patch_size)
            src_patches_rwpe = self.projection_layer['rwpe'](src_patches_rwpe)
            dst_patches_rwpe = self.projection_layer['rwpe'](dst_patches_rwpe)

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

        if self.use_rwpe:
            patches_data = [
                patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features,
                torch.cat([src_patches_rwpe, dst_patches_rwpe], dim=1),
            ]
        else:
            patches_data = [
                patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features,
            ]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)
        patches_data = patches_data.reshape(
            batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        if self.attention_dim_adjust is not None:
            patches_data = self.attention_dim_adjust(patches_data)

        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        if self.attention_dim_revert is not None:
            patches_data = self.attention_dim_revert(patches_data)

        src_patches_data = patches_data[:, : src_num_patches, :]
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def compute_rwpe_features(self, padded_nodes_neighbor_ids: np.ndarray, 
                             nodes_neighbor_ids_list: list, 
                             node_ids: np.ndarray,
                             node_interact_times: np.ndarray):
        """
        为邻居序列计算 RWPE 特征（优化版本）
        利用 NeighborSampler 的 adj_list 构建子图。
        返回前经 float16→float32，与预计算存盘精度一致。

        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param nodes_neighbor_ids_list: list, 每个元素是该节点的邻居ID列表
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return: Tensor, shape (batch_size, max_seq_length, rwpe_dim)
        """
        # 使用 no_grad 避免构建计算图，提升性能
        with torch.no_grad():
            batch_size, max_seq_length = padded_nodes_neighbor_ids.shape
            rwpe_features = torch.zeros(batch_size, max_seq_length, self.rwpe_dim, device=self.device)
            
            for batch_idx in range(batch_size):
                # 获取该样本的节点序列（去除padding）
                node_seq = padded_nodes_neighbor_ids[batch_idx]
                valid_nodes = node_seq[node_seq != 0]  # 去除padding
                
                if len(valid_nodes) < 2:
                    # 如果节点太少，无法构建图，使用零向量
                    continue
                
                # 构建子图：使用当前节点和其邻居
                unique_nodes = np.unique(valid_nodes)
                num_nodes = len(unique_nodes)
                
                # 子图太大则跳过 RWPE（与预计算 max_subgraph_size 一致）
                if num_nodes > self.max_subgraph_size:
                    continue
                
                node_to_local = {int(node): i for i, node in enumerate(unique_nodes)}
                
                # 利用 NeighborSampler 的 adj_list 构建边
                # 获取中心节点
                center_node = int(node_ids[batch_idx])
                # 确保 current_time 是 Python 标量，避免类型不匹配
                current_time_val = node_interact_times[batch_idx]
                if isinstance(current_time_val, torch.Tensor):
                    current_time = float(current_time_val.item())
                elif isinstance(current_time_val, np.ndarray):
                    current_time = float(current_time_val.item() if current_time_val.ndim == 0 else current_time_val[0])
                else:
                    current_time = float(current_time_val)
                
                # 构建边：基于 adj_list 中的邻居关系（只考虑时间戳 <= current_time 的边）
                src_list, dst_list = [], []
                
                # 优化：使用集合查找替代字典查找，提升性能
                unique_nodes_set = set(unique_nodes.astype(int))
                
                # 从 adj_list 中获取所有相关边
                for node_id in unique_nodes:
                    node_id_int = int(node_id)
                    if node_id_int < len(self.neighbor_sampler.nodes_neighbor_ids):
                        # 获取该节点的所有邻居（在时间 current_time 之前）
                        neighbors = self.neighbor_sampler.nodes_neighbor_ids[node_id_int]
                        neighbor_times = self.neighbor_sampler.nodes_neighbor_times[node_id_int]
                        
                        # 确保 neighbor_times 是 numpy 数组
                        if isinstance(neighbor_times, torch.Tensor):
                            neighbor_times = neighbor_times.cpu().numpy()
                        elif not isinstance(neighbor_times, np.ndarray):
                            neighbor_times = np.asarray(neighbor_times)
                        
                        # 优化：使用向量化操作和集合查找
                        if len(neighbors) > 0:
                            # 创建掩码：时间条件
                            time_mask = neighbor_times <= current_time
                            # 创建掩码：节点在unique_nodes中
                            node_mask = np.isin(neighbors, unique_nodes_set)
                            # 组合掩码
                            valid_mask = time_mask & node_mask
                            valid_neighbors = neighbors[valid_mask]
                            
                            if len(valid_neighbors) > 0:
                                local_src = node_to_local[node_id_int]
                                for neighbor_id in valid_neighbors:
                                    neighbor_id_int = int(neighbor_id)
                                    if neighbor_id_int in node_to_local:
                                        src_list.append(local_src)
                                        dst_list.append(node_to_local[neighbor_id_int])
                
                # 去重边（避免重复）- 使用集合更高效
                if len(src_list) > 0:
                    # 使用集合去重
                    edges_set = set(zip(src_list, dst_list))
                    if edges_set:
                        src_list, dst_list = zip(*edges_set)
                        src_list, dst_list = list(src_list), list(dst_list)
                    else:
                        src_list, dst_list = [], []
                    
                    # 使用 SciPy 构建邻接矩阵（替代 dgl）
                    try:
                        # 创建邻接矩阵（CSR 格式）
                        # 注意：这里构建的是有向图，src -> dst
                        A = sp.csr_matrix((np.ones(len(src_list)), (src_list, dst_list)), 
                                         shape=(num_nodes, num_nodes))
                        
                        # 计算入度（替代 dgl.backend.asnumpy(g.in_degrees())）
                        # 入度 = 每列的非零元素个数
                        in_degrees = np.array(A.sum(axis=0)).flatten()  # 按列求和得到入度
                        degrees = np.clip(in_degrees, 1, None)  # 确保至少为1，避免除零
                        
                        # 计算度矩阵的逆
                        Dinv = sp.diags(degrees ** -1.0, dtype=float)
                        
                        # 计算随机游走矩阵 RW = A * D^{-1}
                        RW = A * Dinv
                        
                        # 优化3: 使用更高效的矩阵幂次计算（使用 dot 方法）
                        # 计算 RWPE
                        PE_list = []
                        M_power = RW  # 从 RW 开始
                        
                        for k in range(self.rwpe_dim):
                            # 直接提取对角线，避免转换为密集矩阵
                            diag = M_power.diagonal()
                            # 直接转换为tensor并移到GPU
                            PE_list.append(torch.from_numpy(diag).float().to(self.device))
                            
                            # 如果不是最后一次，计算下一次幂
                            if k < self.rwpe_dim - 1:
                                # 优化4: 使用 dot 方法进行稀疏矩阵乘法，通常比 * 更高效
                                M_power = M_power.dot(RW)
                        
                        PE = torch.stack(PE_list, dim=-1)  # shape: (num_nodes, rwpe_dim)
                        
                        # 将 RWPE 映射回原始序列位置
                        for seq_idx, node_id in enumerate(node_seq):
                            node_id_int = int(node_id)
                            if node_id_int != 0 and node_id_int in node_to_local:
                                local_idx = node_to_local[node_id_int]
                                if local_idx < PE.shape[0]:
                                    rwpe_features[batch_idx, seq_idx] = PE[local_idx]
                    except Exception as e:
                        # 如果计算失败（例如图太小），保持为零向量
                        pass

            # Match precompute storage: float16 quantize then promote back to float32
            return rwpe_features.half().float()

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

    def get_patches_rwpe(self, rwpe_features: torch.Tensor, patch_size: int = 1):
        """
        Patch RWPE features (same pattern as DyGFormerST.get_patches_spatial).
        :param rwpe_features: Tensor, shape (batch_size, max_seq_length, rwpe_dim)
        :param patch_size: int
        :return: Tensor, shape (batch_size, num_patches, patch_size * rwpe_dim)
        """
        assert rwpe_features.shape[1] % patch_size == 0
        num_patches = rwpe_features.shape[1] // patch_size
        batch_size = rwpe_features.shape[0]

        patches = []
        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = start_idx + patch_size
            patches.append(rwpe_features[:, start_idx:end_idx, :])

        return torch.stack(patches, dim=1).reshape(
            batch_size, num_patches, patch_size * self.rwpe_dim
        )

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

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
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

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs