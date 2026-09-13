import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.stable_rank_collector import StableRankCollector


# ===== Metrics helpers: stable rank & spectral/singular norms (no grad) =====
@torch.no_grad()
def _top_singular_value(X: torch.Tensor, n_iter: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Power iteration to approximate σ_max(X). Works for [T,d] or [T,T]."""
    if X.numel() == 0:
        return X.new_tensor(0.0)
    v = torch.randn(X.shape[-1], device=X.device, dtype=X.dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iter):
        v = X.transpose(-1, -2) @ (X @ v)
        v = v / (v.norm() + eps)
    sigma = (X @ v).norm()
    return sigma

@torch.no_grad()
def _stable_rank_xxt(X: torch.Tensor, n_iter: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Stable rank of XX^T: ||X^T X||_F^2 / σ_max(X)^4. X: [T,d] with T>=2."""
    if X.dim() != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        return X.new_tensor(1.0)
    G = X.transpose(-1, -2) @ X              # [d,d]
    fro2 = (G * G).sum()
    smax = _top_singular_value(X, n_iter=n_iter, eps=eps) + eps
    sr = (fro2 / (smax ** 4 + eps)).clamp_min(1.0)
    return sr

@torch.no_grad()
def _attn_gap_norm(A: torch.Tensor, n_iter: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Proxy for spectral gap: ||A - 11^T/T||_2 on a valid submatrix A (row-stochastic)."""
    T = A.shape[0]
    if T <= 1:
        return A.new_tensor(0.0)
    J = A.new_ones((T, T)) / float(T)
    M = A - J
    v = torch.randn(T, device=A.device, dtype=A.dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iter):
        v = M.transpose(-1, -2) @ (M @ v)
        v = v / (v.norm() + eps)
    spec = (M @ v).norm()
    return spec


class DyGFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu',
                 enable_stable_rank_collection: bool = False, stable_rank_save_dir: str = "results/stable_rank_analysis"):
        super(DyGFormer, self).__init__()
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
        
        # 稳定秩收集器
        self.enable_stable_rank_collection = enable_stable_rank_collection
        if self.enable_stable_rank_collection:
            self.stable_rank_collector = StableRankCollector(
                save_dir=stable_rank_save_dir, 
                model_name="DyGFormer"
            )
        else:
            self.stable_rank_collector = None

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim).to(device)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device
        ).to(device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True).to(device),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True).to(device),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True).to(device),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True).to(device)
        })

        self.num_channels = 4

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim,
                               num_heads=self.num_heads, dropout=self.dropout).to(device)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim,
                                      out_features=self.node_feat_dim, bias=True).to(device)

        self.to(device)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = None, 
                                                 record_last_attn: bool = False, **kwargs):
        # 获取 device
        if isinstance(src_node_ids, torch.Tensor):
            device = src_node_ids.device
        elif isinstance(dst_node_ids, torch.Tensor):
            device = dst_node_ids.device
        else:
            device = torch.device('cpu')

        # 获取邻居 - 使用历史采样获取更多邻居
        # 如果num_neighbors未指定，使用较大的默认值以获得更长序列
        # 默认采样200个邻居，确保有足够长的序列进行稳定秩分析
        sample_num_neighbors = num_neighbors if num_neighbors is not None else min(200, self.max_input_sequence_length - 1)
        
        src_nodes_neighbor_ids, src_nodes_edge_ids, src_nodes_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=src_node_ids, 
                node_interact_times=node_interact_times,
                num_neighbors=sample_num_neighbors
            )
        dst_nodes_neighbor_ids, dst_nodes_edge_ids, dst_nodes_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=dst_node_ids, 
                node_interact_times=node_interact_times,
                num_neighbors=sample_num_neighbors
            )
        
        # 将numpy数组转换为列表格式，并移除填充的0
        src_nodes_neighbor_ids_list = []
        src_nodes_edge_ids_list = []
        src_nodes_neighbor_times_list = []
        for i in range(len(src_node_ids)):
            # 找到非零元素（真实的邻居）
            valid_mask = src_nodes_neighbor_ids[i] != 0
            src_nodes_neighbor_ids_list.append(src_nodes_neighbor_ids[i][valid_mask])
            src_nodes_edge_ids_list.append(src_nodes_edge_ids[i][valid_mask])
            src_nodes_neighbor_times_list.append(src_nodes_neighbor_times[i][valid_mask])
        
        dst_nodes_neighbor_ids_list = []
        dst_nodes_edge_ids_list = []
        dst_nodes_neighbor_times_list = []
        for i in range(len(dst_node_ids)):
            valid_mask = dst_nodes_neighbor_ids[i] != 0
            dst_nodes_neighbor_ids_list.append(dst_nodes_neighbor_ids[i][valid_mask])
            dst_nodes_edge_ids_list.append(dst_nodes_edge_ids[i][valid_mask])
            dst_nodes_neighbor_times_list.append(dst_nodes_neighbor_times[i][valid_mask])

        # Padding（保留序列以便生成 patch-level 标签）
        src_pad_ids, src_pad_eids, src_pad_times, src_pad_types = self.pad_sequences(
            node_ids=src_node_ids, node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
            nodes_edge_ids_list=src_nodes_edge_ids_list,
            nodes_neighbor_times_list=src_nodes_neighbor_times_list,
            patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length
        )
        dst_pad_ids, dst_pad_eids, dst_pad_times, dst_pad_types = self.pad_sequences(
            node_ids=dst_node_ids, node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
            nodes_edge_ids_list=dst_nodes_edge_ids_list,
            nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
            patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length
        )

        # 关键节点标记（src序列中与dst相同的邻居；dst序列中与src相同的邻居）
        for i in range(len(src_node_ids)):
            dst_node_id = dst_node_ids[i]
            src_node_id = src_node_ids[i]
            if isinstance(dst_node_id, torch.Tensor):
                dst_node_id = int(dst_node_id.item())
            if isinstance(src_node_id, torch.Tensor):
                src_node_id = int(src_node_id.item())
            src_key_mask = (src_pad_ids[i] == dst_node_id)
            src_pad_types[i][src_key_mask] = 1  # 关键节点
            dst_key_mask = (dst_pad_ids[i] == src_node_id)
            dst_pad_types[i][dst_key_mask] = 1  # 关键节点

        # 邻居共现
        src_nei_co, dst_nei_co = self.neighbor_co_occurrence_encoder(
            src_padded_nodes_neighbor_ids=src_pad_ids,
            dst_padded_nodes_neighbor_ids=dst_pad_ids
        )

        # 特征
        src_node_raw, src_edge_raw, src_time_feat = self.get_features(
            node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_pad_ids,
            padded_nodes_edge_ids=src_pad_eids, padded_nodes_neighbor_times=src_pad_times, time_encoder=self.time_encoder
        )
        dst_node_raw, dst_edge_raw, dst_time_feat = self.get_features(
            node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_pad_ids,
            padded_nodes_edge_ids=dst_pad_eids, padded_nodes_neighbor_times=dst_pad_times, time_encoder=self.time_encoder
        )

        # 组成 patches
        (src_patch_node_raw, src_patch_edge_raw, src_patch_time_feat, src_patch_nei_co) = self.get_patches(
            padded_nodes_neighbor_node_raw_features=src_node_raw, padded_nodes_edge_raw_features=src_edge_raw,
            padded_nodes_neighbor_time_features=src_time_feat, padded_nodes_neighbor_co_occurrence_features=src_nei_co,
            patch_size=self.patch_size
        )
        (dst_patch_node_raw, dst_patch_edge_raw, dst_patch_time_feat, dst_patch_nei_co) = self.get_patches(
            padded_nodes_neighbor_node_raw_features=dst_node_raw, padded_nodes_edge_raw_features=dst_edge_raw,
            padded_nodes_neighbor_time_features=dst_time_feat, padded_nodes_neighbor_co_occurrence_features=dst_nei_co,
            patch_size=self.patch_size
        )

        # Patch-level 关键列标签
        def build_patch_key_mask(padded_ids, target_ids, patch_size):
            B, T = padded_ids.shape
            num_patches = T // patch_size
            ids_t = torch.from_numpy(padded_ids).to(self.device)
            if isinstance(target_ids, torch.Tensor):
                tgt = target_ids.to(self.device).unsqueeze(1)
            else:
                tgt = torch.from_numpy(target_ids).to(self.device).unsqueeze(1)
            ids_patch = ids_t.view(B, num_patches, patch_size)
            key = (ids_patch == tgt.unsqueeze(-1)).any(dim=-1)  # [B, num_patches]
            return key  # bool

        keycols_src = build_patch_key_mask(src_pad_ids, dst_node_ids, self.patch_size)
        keycols_dst = build_patch_key_mask(dst_pad_ids, src_node_ids, self.patch_size)

        # 线性映射到通道维
        src_node = self.projection_layer['node'](src_patch_node_raw)
        src_edge = self.projection_layer['edge'](src_patch_edge_raw)
        src_time = self.projection_layer['time'](src_patch_time_feat)
        src_neic = self.projection_layer['neighbor_co_occurrence'](src_patch_nei_co)

        dst_node = self.projection_layer['node'](dst_patch_node_raw)
        dst_edge = self.projection_layer['edge'](dst_patch_edge_raw)
        dst_time = self.projection_layer['time'](dst_patch_time_feat)
        dst_neic = self.projection_layer['neighbor_co_occurrence'](dst_patch_nei_co)

        B = src_node.shape[0]
        src_num_patches = src_node.shape[1]
        dst_num_patches = dst_node.shape[1]

        # 拼接通道 -> token
        src_tokens = torch.stack([src_node, src_edge, src_time, src_neic], dim=2).reshape(
            B, src_num_patches, self.num_channels * self.channel_embedding_dim
        )
        dst_tokens = torch.stack([dst_node, dst_edge, dst_time, dst_neic], dim=2).reshape(
            B, dst_num_patches, self.num_channels * self.channel_embedding_dim
        )
        tokens = torch.cat([src_tokens, dst_tokens], dim=1)  # [B, T_total, C]

        # Transformer 层（仅在最后一层记录注意力权重与注意力输出），并收集稳定秩信息
        last_attn = None
        h = tokens
        layer_embeddings = []  # 存储每层的嵌入
        layer_attentions = []  # 存储每层的注意力权重
        
        # 只在启用稳定秩收集时才收集每层数据
        collect_layer_data = self.enable_stable_rank_collection
        
        for li, encoder in enumerate(self.transformers):
            record = (li == len(self.transformers) - 1)
            h, attn = encoder(h, record_attn=record,
                               node_categories=np.concatenate([src_pad_types, dst_pad_types], axis=1) if record else None)
            
            # 只在需要时收集每层的嵌入和注意力权重
            if collect_layer_data:
                # 只收集第一个batch的数据，减少内存使用
                layer_embeddings.append(h[0:1].clone().detach())  # 只保留第一个batch
                if attn is not None and 'A' in attn:
                    layer_attentions.append(attn['A'][0:1].clone().detach())  # 只保留第一个batch
                else:
                    layer_attentions.append(None)
            else:
                # 不收集时，只保留最后一层用于基本功能
                if li == len(self.transformers) - 1:
                    layer_embeddings = [h[0:1].clone().detach()]
                    if attn is not None and 'A' in attn:
                        layer_attentions = [attn['A'][0:1].clone().detach()]
                    else:
                        layer_attentions = [None]
            
            if record:
                last_attn = attn

        # 平均池化到节点向量
        src_feat = torch.mean(h[:, :src_num_patches, :], dim=1)
        dst_feat = torch.mean(h[:, src_num_patches:, :], dim=1)

        src_emb = self.output_layer(src_feat)
        dst_emb = self.output_layer(dst_feat)

        # --- 切出 src/dst 对齐的注意力子矩阵 + 构造标签矩阵 ---
        extra = {}
        if last_attn is not None and ('A' in last_attn):
            A_full = last_attn['A']  # [B, H, T_total, T_total]（假定返回已 reshape）
            H = A_full.shape[1]
            # src / dst 视角
            A_src = A_full[:, :, :src_num_patches, :src_num_patches]
            A_dst = A_full[:, :, src_num_patches:, src_num_patches:]

            # patch-level 标签矩阵（列为 key 维）
            src_types = torch.from_numpy(src_pad_types).to(self.device)
            dst_types = torch.from_numpy(dst_pad_types).to(self.device)

            def node_to_patch_labels(node_labels, num_patches, patch_size):
                B, T = node_labels.shape
                patch_labels = torch.zeros(B, num_patches, dtype=torch.float32, device=self.device)
                for b in range(B):
                    for p in range(num_patches):
                        start_idx = p * patch_size
                        end_idx = min((p + 1) * patch_size, T)
                        patch_node_labels = node_labels[b, start_idx:end_idx]
                        if (patch_node_labels == 1).any():
                            patch_labels[b, p] = 1    # 关键
                        elif (patch_node_labels == 2).any():
                            patch_labels[b, p] = 2    # 次要关键
                        elif (patch_node_labels == 0).any():
                            patch_labels[b, p] = 0    # self
                        elif (patch_node_labels == 4).any():
                            patch_labels[b, p] = 4    # padding
                        else:
                            patch_labels[b, p] = 3    # other
                return patch_labels

            src_patch_labels = node_to_patch_labels(src_types, src_num_patches, self.patch_size)
            dst_patch_labels = node_to_patch_labels(dst_types, dst_num_patches, self.patch_size)

            Bsz, H_, src_T, _ = A_src.shape
            _, _, dst_T, _ = A_dst.shape
            node_labels_src = torch.zeros(Bsz, H_, src_T, src_T, dtype=torch.float32, device=self.device)
            node_labels_dst = torch.zeros(Bsz, H_, dst_T, dst_T, dtype=torch.float32, device=self.device)
            for b in range(Bsz):
                for h_i in range(H_):
                    for col in range(src_T):
                        if col < src_patch_labels.shape[1]:
                            node_labels_src[b, h_i, :, col] = src_patch_labels[b, col]
                    for col in range(dst_T):
                        if col < dst_patch_labels.shape[1]:
                            node_labels_dst[b, h_i, :, col] = dst_patch_labels[b, col]

            # 注意力输出（来自最后一层 MHA 的 Y），用于在 Y/Y_perp 上统计稳定秩
            Y_full = last_attn.get('Y', None)  # [B, T_total, C] 或 None

            extra = {
                "attn_weights": {"src": A_src, "dst": A_dst},
                "attn_scores":  {"src": {"A": A_src}, "dst": {"A": A_dst}},
                "node_labels":  {"src": node_labels_src, "dst": node_labels_dst},
                "meta": {"src_len": int(src_num_patches), "dst_len": int(dst_num_patches)},
                "attn_out": Y_full
            }

                        # 暂时注释掉注意力权重分析
            # attention_analysis = self._analyze_attention_weights(src_pad_types, dst_pad_types)
            # if attention_analysis:
            #     extra["attn_weights"].update(attention_analysis)
        
        # ================= 每层稳定秩统计 (参考论文实现) =================
        try:
            with torch.no_grad():
                layer_stable_ranks = {"src": [], "dst": []}
                
                # 收集每一层的token embeddings
                layer_embeddings = []
                
                # 重新前向传播收集每层embeddings
                h_temp = tokens.clone()
                
                for i, encoder in enumerate(self.transformers):
                    # 前向传播但不记录注意力
                    h_temp, _ = encoder(h_temp, record_attn=False)
                    
                    # 收集当前层的embeddings
                    layer_embeddings.append(h_temp.clone())
                
                # 计算每层的稳定秩
                for l, layer_emb in enumerate(layer_embeddings):
                    # SRC
                    src_stable_ranks = []
                    for b in range(layer_emb.shape[0]):
                        # 使用patch级别的有效mask
                        vm = src_patch_valid_mask[b] if 'src_patch_valid_mask' in locals() else torch.from_numpy(src_pad_types[b] != 4).to(self.device)
                        if vm.sum() >= 2:  # 至少需要2个有效patch
                            X = layer_emb[b, :src_num_patches, :][vm]  # [T_valid, E]
                            try:
                                sigma = X @ X.T  # [T_valid, T_valid]
                                s = torch.linalg.svdvals(sigma)
                                stable_rank = ((s**2).sum() / (s**2).max()).item()
                                src_stable_ranks.append(stable_rank)
                            except:
                                src_stable_ranks.append(float('nan'))
                    
                    # DST
                    dst_stable_ranks = []
                    for b in range(layer_emb.shape[0]):
                        # 使用patch级别的有效mask
                        vm = dst_patch_valid_mask[b] if 'dst_patch_valid_mask' in locals() else torch.from_numpy(dst_pad_types[b] != 4).to(self.device)
                        if vm.sum() >= 2:  # 至少需要2个有效patch
                            X = layer_emb[b, src_num_patches:, :][vm]  # [T_valid, E]
                            try:
                                sigma = X @ X.T  # [T_valid, T_valid]
                                s = torch.linalg.svdvals(sigma)
                                stable_rank = ((s**2).sum() / (s**2).max()).item()
                                dst_stable_ranks.append(stable_rank)
                            except:
                                dst_stable_ranks.append(float('nan'))
                    
                    layer_stable_ranks["src"].append({
                        "layer": l,
                        "stable_ranks": src_stable_ranks,
                        "mean": np.nanmean(src_stable_ranks) if src_stable_ranks else float('nan'),
                        "std": np.nanstd(src_stable_ranks) if src_stable_ranks else float('nan')
                    })
                    
                    layer_stable_ranks["dst"].append({
                        "layer": l,
                        "stable_ranks": dst_stable_ranks,
                        "mean": np.nanmean(dst_stable_ranks) if dst_stable_ranks else float('nan'),
                        "std": np.nanstd(dst_stable_ranks) if dst_stable_ranks else float('nan')
                    })
                
                extra["layer_stable_ranks"] = layer_stable_ranks
        except Exception as _e:
            extra["layer_stable_ranks_error"] = str(_e)
        
        # ================= 基于StableRankCollector的稳定秩计算（与stable_rank_collector.py完全一致）=================
        try:
            with torch.no_grad():
                metrics = {"src": [], "dst": []}
                
                # 只在启用稳定秩收集且有必要数据时才计算
                if (B > 0 and layer_embeddings and 
                    self.enable_stable_rank_collection and 
                    len(layer_embeddings) > 0):
                    
                    # 只对最后一层计算稳定秩，大幅减少计算量
                    last_layer_idx = len(layer_embeddings) - 1
                    layer_emb = layer_embeddings[last_layer_idx]
                    
                    # 分离src和dst部分（取第一个batch样本）
                    src_emb = layer_emb[0, :src_num_patches, :]  # [src_patches, E] 第一个batch的src部分
                    dst_emb = layer_emb[0, src_num_patches:, :]  # [dst_patches, E] 第一个batch的dst部分
                    
                    # 计算src的稳定秩（使用与stable_rank_collector.py完全相同的方法）
                    if src_emb.shape[0] >= 2:
                        try:
                            # 使用stable_rank_collector.py的compute_stable_rank方法（xxt方法）
                            X = src_emb  # [T, d]
                            
                            # XX^T 方法：||X^T X||_F^2 / σ_max(X)^4
                            G = X.transpose(-1, -2) @ X  # [d, d]
                            fro_norm_sq = (G * G).sum()
                            
                            # 计算最大奇异值
                            _, s, _ = torch.svd(X)
                            sigma_max = s.max()
                            
                            stable_rank = (fro_norm_sq / (sigma_max ** 4 + 1e-8)).clamp_min(1.0).item()
                            
                            # 计算谱间隙（使用与stable_rank_collector.py相同的方法）
                            gap = None
                            if (last_attn is not None) and ("A" in last_attn) and (last_attn["A"] is not None):
                                A = last_attn["A"][0, :, :src_num_patches, :src_num_patches]   # [H, Ts, Ts]
                                A = A.mean(dim=0)  # [Ts, Ts] 头均值
                                # 使用stable_rank_collector.py的compute_spectral_gap方法
                                T = A.shape[0]
                                if T > 1:
                                    J = torch.ones_like(A) / float(T)
                                    M = A - J
                                    _, s_gap, _ = torch.svd(M)
                                    gap = float(s_gap.max())
                            
                            metrics["src"].append({
                                "layer": last_layer_idx,
                                "T_valid": src_emb.shape[0],
                                "stable_rank": stable_rank,
                                "gap_norm": gap,
                            })
                        except Exception as e:
                            # print(f"Warning: SRC stable rank calculation failed: {e}")
                            pass
                    
                    # 计算dst的稳定秩（使用与stable_rank_collector.py完全相同的方法）
                    if dst_emb.shape[0] >= 2:
                        try:
                            # 使用stable_rank_collector.py的compute_stable_rank方法（xxt方法）
                            X = dst_emb  # [T, d]
                            
                            # XX^T 方法：||X^T X||_F^2 / σ_max(X)^4
                            G = X.transpose(-1, -2) @ X  # [d, d]
                            fro_norm_sq = (G * G).sum()
                            
                            # 计算最大奇异值
                            _, s, _ = torch.svd(X)
                            sigma_max = s.max()
                            
                            stable_rank = (fro_norm_sq / (sigma_max ** 4 + 1e-8)).clamp_min(1.0).item()
                            
                            # 计算谱间隙（使用与stable_rank_collector.py相同的方法）
                            gap = None
                            if (last_attn is not None) and ("A" in last_attn) and (last_attn["A"] is not None):
                                A = last_attn["A"][0, :, src_num_patches:, src_num_patches:]   # [H, Td, Td]
                                A = A.mean(dim=0)  # [Td, Td] 头均值
                                # 使用stable_rank_collector.py的compute_spectral_gap方法
                                T = A.shape[0]
                                if T > 1:
                                    J = torch.ones_like(A) / float(T)
                                    M = A - J
                                    _, s_gap, _ = torch.svd(M)
                                    gap = float(s_gap.max())
                            
                            metrics["dst"].append({
                                "layer": last_layer_idx,
                                "T_valid": dst_emb.shape[0],
                                "stable_rank": stable_rank,
                                "gap_norm": gap,
                            })
                        except Exception as e:
                            # print(f"Warning: DST stable rank calculation failed: {e}")
                            pass

                extra["metrics"] = metrics
        except Exception as _e:
            extra["metrics_error"] = str(_e)
        
        # ================= 新增：稳定秩收集（无条件执行，如果启用的话）=================
        # 添加metrics数据（与DIFFDyG保持一致）
        try:
            if layer_embeddings and layer_attentions:
                metrics = {"src": [], "dst": []}
                
                # 只处理最后一层的注意力权重
                last_layer_idx = len(layer_embeddings) - 1
                last_h = layer_embeddings[last_layer_idx]  # [B, T_total, C]
                last_attn = layer_attentions[last_layer_idx] if last_layer_idx < len(layer_attentions) else None
                
                # 分离src和dst
                src_h = last_h[:, :src_num_patches, :]  # [B, src_patches, C]
                dst_h = last_h[:, src_num_patches:, :]  # [B, dst_patches, C]
                
                # 计算稳定秩
                for side, h_side in [("src", src_h), ("dst", dst_h)]:
                    if h_side.shape[1] >= 2:  # 至少需要2个序列元素
                        # 取第一个批次样本进行分析
                        X = h_side[0]  # [T, d]
                        T_valid = X.shape[0]
                        
                        # 计算稳定秩
                        with torch.no_grad():
                            sr_val = _stable_rank_xxt(X).item()
                            
                            # 去中心化
                            X_centered = X - X.mean(dim=0, keepdim=True)
                            sr_centered = _stable_rank_xxt(X_centered).item()
                            
                            # 去均匀投影
                            ones = torch.ones(T_valid, 1, device=X.device) / math.sqrt(T_valid)
                            X_perp = X - ones @ (ones.T @ X)
                            sr_perp = _stable_rank_xxt(X_perp).item()
                        
                        metrics[side].append({
                            "T_valid": T_valid,
                            "sr_xxt": sr_val,
                            "sr_xxt_centered": sr_centered,
                            "sr_xxt_perp": sr_perp,
                            "gap_norm": 0.0,  # DyGFormer不计算谱间隙
                            "gap_norm_perp": 0.0
                        })
                
                extra["metrics"] = metrics
        except Exception as e:
            extra["metrics_error"] = str(e)
        
        if self.enable_stable_rank_collection and self.stable_rank_collector is not None:
            try:
                # 只收集最后一层的稳定秩指标以提高性能
                last_layer_idx = len(layer_embeddings) - 1
                
                # 分别处理src和dst部分
                layer_emb = layer_embeddings[last_layer_idx]
                layer_attn = layer_attentions[last_layer_idx] if last_layer_idx < len(layer_attentions) else None
                
                # SRC层级指标
                src_emb = layer_emb[:, :src_num_patches, :]  # [B, src_patches, C]
                src_attn = None
                if layer_attn is not None:
                    src_attn = layer_attn[:, :, :src_num_patches, :src_num_patches]  # [B, H, src_patches, src_patches]
                
                src_metrics = self.stable_rank_collector.collect_batch_metrics(
                    embeddings=src_emb,
                    attention_weights=src_attn,
                    layer_idx=last_layer_idx,
                    batch_idx=getattr(self, '_batch_idx', 0),
                    node_types=src_pad_types
                )
                src_metrics['data_type'] = 'src'
                self.stable_rank_collector.add_layer_data([src_metrics])
                
                # DST层级指标
                dst_emb = layer_emb[:, src_num_patches:, :]  # [B, dst_patches, C]
                dst_attn = None
                if layer_attn is not None:
                    dst_attn = layer_attn[:, :, src_num_patches:, src_num_patches:]  # [B, H, dst_patches, dst_patches]
                
                dst_metrics = self.stable_rank_collector.collect_batch_metrics(
                    embeddings=dst_emb,
                    attention_weights=dst_attn,
                    layer_idx=last_layer_idx,
                    batch_idx=getattr(self, '_batch_idx', 0),
                    node_types=dst_pad_types
                )
                dst_metrics['data_type'] = 'dst'
                self.stable_rank_collector.add_layer_data([dst_metrics])
                
                # 增加批次计数器
                if not hasattr(self, '_batch_idx'):
                    self._batch_idx = 0
                else:
                    self._batch_idx += 1
                    
            except Exception as e:
                extra["stable_rank_collection_error"] = str(e)
                # 回退：用最后一层表示快速估计一份稳定秩，保证至少有数据
                try:
                    with torch.no_grad():
                        if 'meta' in extra and isinstance(extra['meta'], dict):
                            src_len = int(extra['meta'].get('src_len', src_num_patches))
                        else:
                            src_len = src_num_patches
                        last = h  # [B,T_total,C]
                        src_last = last[:, :src_len, :]
                        dst_last = last[:, src_len:, :]

                        def _quick_sr(X: torch.Tensor) -> float:
                            # X: [B,T,d] -> use first sample valid tokens if exist
                            if X.dim() != 3 or X.shape[1] < 2:
                                return float('nan')
                            X0 = X[0]
                            G = X0.transpose(-1, -2) @ X0
                            fro2 = float((G * G).sum().item())
                            # spectral norm approx by max row norm (cheap fallback)
                            row_norms = torch.linalg.norm(X0, dim=-1)
                            smax2 = float((row_norms.max() ** 2).item()) + 1e-8
                            return max(1.0, fro2 / (smax2 * smax2 + 1e-8))

                        fallback = {
                            "src": [{"layer": len(layer_embeddings) - 1, "stable_ranks": [], "mean": _quick_sr(src_last), "std": 0.0}],
                            "dst": [{"layer": len(layer_embeddings) - 1, "stable_ranks": [], "mean": _quick_sr(dst_last), "std": 0.0}],
                        }
                        extra.setdefault("layer_stable_ranks", fallback)
                except Exception:
                    pass
        elif self.enable_stable_rank_collection:
            # 调试信息：为什么没有收集到数据
            debug_info = {
                "has_collector": self.stable_rank_collector is not None,
                "has_embeddings": bool(layer_embeddings),
                "num_embeddings": len(layer_embeddings) if layer_embeddings else 0,
                "enabled": self.enable_stable_rank_collection,
                "num_transformers": len(self.transformers),
                "layer_embeddings_collected": len(layer_embeddings) if layer_embeddings else 0
            }
            extra["stable_rank_debug"] = debug_info
            # print(f"DyGFormer 调试: {debug_info}")  # 临时调试输出

        # 计算稳定秩和谱间隙信息（不打印，只在需要时计算）
        # 暂时注释掉，避免每个batch都计算
        # if last_attn is not None and 'A' in last_attn:
        #     self._compute_metrics(h, last_attn, src_num_patches, dst_num_patches)

        return src_emb, dst_emb, extra

        # 暂时注释掉注意力权重分析方法
    # def _analyze_attention_weights(self, src_categories, dst_categories):
    #     """
    #     分析注意力权重，统计四类节点的注意力分配占比
    #     """
    #     extra_info = {}
    #     for i, encoder in enumerate(self.transformers):
    #         if hasattr(encoder, 'last_attn_weights') and encoder.last_attn_weights is not None:
    #             attn_weights = encoder.last_attn_weights  # 期望形状 [B,H,T,T]
    #             node_categories = encoder.node_categories
    #             if attn_weights is not None and node_categories is not None:
    #                 for side, categories in [('src', src_categories), ('dst', dst_categories)]:
    #                     if categories is not None:
    #                         B, H, T, _ = attn_weights.shape
    #                         if categories.shape[1] != T:
    #                             categories = categories[:, :T] if categories.shape[1] > T else categories
    #                         category_weights = torch.zeros(4, device=attn_weights.device)  # [self, key, other, pad]
    #                         total_attn = torch.sum(attn_weights, dim=(1, 2))  # [B, T]
    #                         for b in range(B):
    #                             for key_idx in range(min(categories.shape[1], total_attn.shape[1])):
    #                                     key_cat = int(categories[b, key_idx])
    #                                 if key_cat == 0:      mapped_cat = 0  # self
    #                                 elif key_cat in [1,2]:mapped_cat = 1  # key/secondary key
    #                                 elif key_cat == 4:    mapped_cat = 3  # pad
    #                                 else:                 mapped_cat = 2  # other
    #                                         category_weights[mapped_cat] += total_attn[b, key_idx].item()
    #                         total_weight = category_weights.sum()
    #                         if total_weight > 0:
    #                             category_weights = category_weights / total_weight
    #                         share_dict = {
    #                             'self': category_weights[0].item(),
    #                             'key': category_weights[1].item(),
    #                             'other': category_weights[2].item(),
    #                             'pad': category_weights[3].item()
    #                         }
    #                         if side not in extra_info:
    #                             extra_info[side] = {}
    #                         extra_info[side]['A'] = {'share': share_dict}
    #             break  # 只处理最后一层
    #     return extra_info

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list,
                      nodes_edge_ids_list: list, nodes_neighbor_times_list: list,
                      patch_size: int = 1, max_input_sequence_length: int = 256):
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            current_len = len(nodes_neighbor_ids_list[idx])
            if current_len > max_seq_length:
                max_seq_length = current_len
        max_seq_length += 1
        max_seq_length = ((max_seq_length + patch_size - 1) // patch_size) * patch_size
        max_seq_length = min(max_seq_length, max_input_sequence_length)

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length), dtype=np.float32)
        # 节点类型标记 (0=self, 1=关键, 2=次要关键, 3=无关, 4=padding)
        padded_node_types = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        
        for idx in range(len(node_ids)):
            # 位置0：self token
            nid = node_ids[idx]
            ts = node_interact_times[idx]
            if isinstance(nid, torch.Tensor): nid = int(nid.item())
            if isinstance(ts, torch.Tensor):  ts = float(ts.item())
            padded_nodes_neighbor_ids[idx, 0] = nid
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = ts
            padded_node_types[idx, 0] = 0  # self token
            
            if len(nodes_neighbor_ids_list[idx]) > 0:
                neighbor_len = min(len(nodes_neighbor_ids_list[idx]), max_seq_length - 1)
                if neighbor_len < len(nodes_neighbor_ids_list[idx]):
                    nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-neighbor_len:]
                    nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-neighbor_len:]
                    nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-neighbor_len:]
                
                # 填充邻居信息
                padded_nodes_neighbor_ids[idx, 1:1 + neighbor_len] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1:1 + neighbor_len] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1:1 + neighbor_len] = nodes_neighbor_times_list[idx]
                
                # 标记节点类型（1-1+neighbor_len位置）
                padded_node_types[idx, 1:1 + neighbor_len] = 3  # 无关节点
                secondary_key_count = max(1, int(neighbor_len * 0.1))
                padded_node_types[idx, 1:1 + secondary_key_count] = 2  # 次要关键节点
                padded_node_types[idx, 1 + neighbor_len:] = 4  # padding
        
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times, padded_node_types

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray, padded_nodes_neighbor_times: np.ndarray,
                     time_encoder: TimeEncoder):
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times_np = node_interact_times.detach().cpu().numpy()
        else:
            node_interact_times_np = node_interact_times
        if isinstance(padded_nodes_neighbor_times, torch.Tensor):
            padded_nodes_neighbor_times_np = padded_nodes_neighbor_times.detach().cpu().numpy()
        else:
            padded_nodes_neighbor_times_np = padded_nodes_neighbor_times
        time_deltas = node_interact_times_np[:, np.newaxis] - padded_nodes_neighbor_times_np
        time_deltas_tensor = torch.from_numpy(time_deltas).float().to(self.device)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=time_deltas_tensor)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids).to(self.device)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids).to(self.device)]
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0).to(self.device)] = 0.0
        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None,
                    patch_size: int = 1):
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

        B = len(padded_nodes_neighbor_node_raw_features)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(
            B, num_patches, patch_size * self.node_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(
            B, num_patches, patch_size * self.edge_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(
            B, num_patches, patch_size * self.time_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(
            B, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return (patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def enable_stable_rank_collection_mode(self, enable: bool = True):
        """启用或禁用稳定秩收集模式"""
        self.enable_stable_rank_collection = enable
        if enable and self.stable_rank_collector is None:
            self.stable_rank_collector = StableRankCollector(
                save_dir="results/stable_rank_analysis", 
                model_name="DyGFormer"
            )
    
    def save_stable_rank_results(self, suffix: str = ""):
        """保存稳定秩收集结果"""
        if self.stable_rank_collector is not None:
            self.stable_rank_collector.save_results(suffix=suffix)
    
    def get_stable_rank_analysis(self, min_sequence_length: int = 10):
        """获取稳定秩分析结果"""
        if self.stable_rank_collector is not None:
            return self.stable_rank_collector.analyze_rank_collapse(min_sequence_length=min_sequence_length)
        return None
    
    def clear_stable_rank_data(self):
        """清空稳定秩收集数据"""
        if self.stable_rank_collector is not None:
            self.stable_rank_collector.clear_data()

    def _compute_metrics(self, h, last_attn, src_num_patches, dst_num_patches):
        """计算稳定秩、谱间隙、秩-1占比等指标（不打印）"""
        try:
            with torch.no_grad():
                # 获取注意力权重
                A_full = last_attn['A']  # [B, H, T_total, T_total]
                H = A_full.shape[1]
                
                # 分离src和dst的注意力权重
                A_src = A_full[:, :, :src_num_patches, :src_num_patches]  # [B, H, src_T, src_T]
                A_dst = A_full[:, :, src_num_patches:, src_num_patches:]  # [B, H, dst_T, dst_T]
                
                # 分离src和dst的嵌入
                src_emb = h[:, :src_num_patches, :]  # [B, src_T, C]
                dst_emb = h[:, src_num_patches:, :]  # [B, dst_T, C]
                
                # 计算src的指标
                src_metrics = self._compute_side_metrics(src_emb, A_src, "src")
                
                # 计算dst的指标
                dst_metrics = self._compute_side_metrics(dst_emb, A_dst, "dst")
                
                # 存储指标到实例变量中，供后续打印使用
                self._last_src_metrics = src_metrics
                self._last_dst_metrics = dst_metrics
                
        except Exception as e:
            # 静默处理错误，不打印
            pass

    def print_epoch_metrics(self):
        """打印epoch结束时的稳定秩和谱间隙指标"""
        if hasattr(self, '_last_src_metrics') and hasattr(self, '_last_dst_metrics'):
            src_metrics = self._last_src_metrics
            dst_metrics = self._last_dst_metrics
            
            # 注释掉打印语句
            # print(f"DyGFormer src 协方差矩阵: 稳定秩={src_metrics['stable_rank']:.4f}, 谱间隙={src_metrics['spectral_gap']:.4f}, 秩-1占比={src_metrics['rank_1_ratio']:.4f}")
            # print(f"DyGFormer dst 协方差矩阵: 稳定秩={dst_metrics['stable_rank']:.4f}, 谱间隙={dst_metrics['spectral_gap']:.4f}, 秩-1占比={dst_metrics['rank_1_ratio']:.4f}")
            # print(f"DyGFormer src 行中心化注意力协方差: 稳定秩={src_metrics['centered_stable_rank']:.4f}, 谱间隙={src_metrics['centered_spectral_gap']:.4f}, 秩-1占比={src_metrics['centered_rank_1_ratio']:.4f}, sr/T={src_metrics['sr_over_T']:.4f}")
            # print(f"DyGFormer dst 行中心化注意力协方差: 稳定秩={dst_metrics['centered_stable_rank']:.4f}, 谱间隙={dst_metrics['centered_spectral_gap']:.4f}, 秩-1占比={dst_metrics['centered_rank_1_ratio']:.4f}, sr/T={dst_metrics['sr_over_T']:.4f}")

    def _compute_and_print_metrics_from_extra(self, extra):
        """从extra数据计算并打印稳定秩指标"""
        try:
            if extra is None or 'metrics' not in extra:
                return
                
            metrics = extra['metrics']
            if 'src' not in metrics or 'dst' not in metrics:
                return
                
            # 计算src的平均指标
            src_data = metrics['src']
            if src_data:
                src_stable_ranks = [d.get('stable_rank', 1.0) for d in src_data if 'stable_rank' in d]
                src_gaps = [d.get('gap_norm', 0.0) for d in src_data if 'gap_norm' in d and d['gap_norm'] is not None]
                src_rank_1_ratios = [d.get('rank_1_ratio', 1.0) for d in src_data if 'rank_1_ratio' in d]
                
                avg_src_sr = np.mean(src_stable_ranks) if src_stable_ranks else 1.0
                avg_src_gap = np.mean(src_gaps) if src_gaps else 0.0
                avg_src_rank_1 = np.mean(src_rank_1_ratios) if src_rank_1_ratios else 1.0
            else:
                avg_src_sr = avg_src_gap = avg_src_rank_1 = 1.0
                
            # 计算dst的平均指标
            dst_data = metrics['dst']
            if dst_data:
                dst_stable_ranks = [d.get('stable_rank', 1.0) for d in dst_data if 'stable_rank' in d]
                dst_gaps = [d.get('gap_norm', 0.0) for d in dst_data if 'gap_norm' in d and d['gap_norm'] is not None]
                dst_rank_1_ratios = [d.get('rank_1_ratio', 1.0) for d in dst_data if 'rank_1_ratio' in d]
                
                avg_dst_sr = np.mean(dst_stable_ranks) if dst_stable_ranks else 1.0
                avg_dst_gap = np.mean(dst_gaps) if dst_gaps else 0.0
                avg_dst_rank_1 = np.mean(dst_rank_1_ratios) if dst_rank_1_ratios else 1.0
            else:
                avg_dst_sr = avg_dst_gap = avg_dst_rank_1 = 1.0
                
            # 注释掉打印结果
            # print(f"DyGFormer src 协方差矩阵: 稳定秩={avg_src_sr:.4f}, 谱间隙={avg_src_gap:.4f}, 秩-1占比={avg_src_rank_1:.4f}")
            # print(f"DyGFormer dst 协方差矩阵: 稳定秩={avg_dst_sr:.4f}, 谱间隙={avg_dst_gap:.4f}, 秩-1占比={avg_dst_rank_1:.4f}")
            # print(f"DyGFormer src 行中心化注意力协方差: 稳定秩={avg_src_sr:.4f}, 谱间隙={avg_src_gap:.4f}, 秩-1占比={avg_src_rank_1:.4f}, sr/T={avg_src_sr/20:.4f}")
            # print(f"DyGFormer dst 行中心化注意力协方差: 稳定秩={avg_dst_sr:.4f}, 谱间隙={avg_dst_gap:.4f}, 秩-1占比={avg_dst_rank_1:.4f}, sr/T={avg_dst_sr/20:.4f}")
            
        except Exception as e:
            pass  # 静默处理错误

    def _compute_side_metrics(self, embeddings, attention_weights, side_name):
        """计算单侧的稳定秩、谱间隙、秩-1占比等指标"""
        try:
            with torch.no_grad():
                # 取第一个batch的数据
                emb = embeddings[0]  # [T, C]
                attn = attention_weights[0]  # [H, T, T]
                
                T = emb.shape[0]
                if T < 2:
                    return self._get_default_metrics()
                
                # 计算协方差矩阵的稳定秩
                cov_matrix = emb @ emb.T  # [T, T]
                stable_rank = self._compute_stable_rank(cov_matrix)
                
                # 计算谱间隙
                spectral_gap = self._compute_spectral_gap(emb)
                
                # 计算秩-1占比
                rank_1_ratio = self._compute_rank_1_ratio(cov_matrix)
                
                # 计算行中心化后的指标
                centered_emb = emb - emb.mean(dim=0, keepdim=True)
                centered_cov = centered_emb @ centered_emb.T
                centered_stable_rank = self._compute_stable_rank(centered_cov)
                centered_spectral_gap = self._compute_spectral_gap(centered_emb)
                centered_rank_1_ratio = self._compute_rank_1_ratio(centered_cov)
                
                # 计算sr/T
                sr_over_T = centered_stable_rank / T
                
                return {
                    'stable_rank': stable_rank,
                    'spectral_gap': spectral_gap,
                    'rank_1_ratio': rank_1_ratio,
                    'centered_stable_rank': centered_stable_rank,
                    'centered_spectral_gap': centered_spectral_gap,
                    'centered_rank_1_ratio': centered_rank_1_ratio,
                    'sr_over_T': sr_over_T
                }
                
        except Exception as e:
            # 静默处理错误，不打印
            return self._get_default_metrics()

    def _compute_stable_rank(self, matrix):
        """计算稳定秩"""
        try:
            # 使用SVD计算稳定秩
            U, s, V = torch.svd(matrix)
            frobenius_norm_sq = (s ** 2).sum()
            spectral_norm_sq = s[0] ** 2
            stable_rank = frobenius_norm_sq / (spectral_norm_sq + 1e-8)
            return stable_rank.item()
        except:
            return 1.0

    def _compute_spectral_gap(self, embeddings):
        """计算谱间隙"""
        try:
            # 计算协方差矩阵的最大特征值
            cov_matrix = embeddings @ embeddings.T
            eigenvals = torch.linalg.eigvals(cov_matrix).real
            max_eigenval = eigenvals.max()
            return max_eigenval.item()
        except:
            return 0.0

    def _compute_rank_1_ratio(self, matrix):
        """计算秩-1占比"""
        try:
            U, s, V = torch.svd(matrix)
            rank_1_ratio = (s[0] ** 2) / ((s ** 2).sum() + 1e-8)
            return rank_1_ratio.item()
        except:
            return 1.0

    def _get_default_metrics(self):
        """返回默认指标值"""
        return {
            'stable_rank': 1.0,
            'spectral_gap': 0.0,
            'rank_1_ratio': 1.0,
            'centered_stable_rank': 1.0,
            'centered_spectral_gap': 0.0,
            'centered_rank_1_ratio': 1.0,
            'sr_over_T': 0.0
        }

    def forward(self, batch):
        src_node_ids = batch['src_node_ids']
        dst_node_ids = batch['dst_node_ids']
        node_interact_times = batch['node_interact_times']
        src_emb, dst_emb, extra = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times
        )
        return src_emb, dst_emb, extra


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
            src_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
            src_mapping = dict(zip(src_unique_keys, src_counts))

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
            dst_mapping = dict(zip(dst_unique_keys, dst_counts))

            src_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(lambda nid: dst_mapping.get(nid, 0.0)).float().to(self.device)
            dst_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(lambda nid: src_mapping.get(nid, 0.0)).float().to(self.device)

            src_padded_nodes_appearances.append(torch.stack([src_counts_in_src, src_in_dst], dim=1))
            dst_padded_nodes_appearances.append(torch.stack([dst_in_src, dst_counts_in_dst], dim=1))

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0
        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_app, dst_app = self.count_nodes_appearances(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids)
        src_feat = self.neighbor_co_occurrence_encode_layer(src_app.unsqueeze(dim=-1)).sum(dim=2)
        dst_feat = self.neighbor_co_occurrence_encode_layer(dst_app.unsqueeze(dim=-1)).sum(dim=2)
        return src_feat, dst_feat


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout, batch_first=False
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
        self.last_attn_weights = None
        self.last_attn_out = None
        self.node_categories = None

    def forward(self, inputs: torch.Tensor, record_attn: bool = False, node_categories=None):
        # inputs: [B, T, C]
        x = inputs.transpose(0, 1)                    # [T, B, C]
        x_norm = self.norm_layers[0](x)
        attn_out, attn_weights = self.multi_head_attention(
            query=x_norm, key=x_norm, value=x_norm,
            need_weights=True, average_attn_weights=False
        )                                              # attn_out: [T,B,C]
        attn_out_btc = attn_out.transpose(0, 1)        # [B,T,C]  —— 注意力输出（用于稳定秩）

        hidden = attn_out_btc
        outputs = inputs + self.dropout(hidden)

        ff_in = self.norm_layers[1](outputs)
        hidden2 = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](ff_in))))
        outputs = outputs + self.dropout(hidden2)

        # 保存注意力权重 + 注意力输出（供指标统计）
        if record_attn and attn_weights is not None:
            # 说明：在部分 PyTorch 版本中，attn_weights 形状可能为 [B*H, T, T]，这里假定外部已处理成 [B, H, T, T]
            self.last_attn_weights = attn_weights
            self.last_attn_out = attn_out_btc
            self.node_categories = node_categories

        attn_dict = None
        if record_attn:
            # 确保注意力权重形状正确 [B, H, T, T]
            if attn_weights is not None:
                if attn_weights.dim() == 3:  # [B*H, T, T]
                    B, T, _ = inputs.shape
                    H = self.multi_head_attention.num_heads
                    attn_weights = attn_weights.view(B, H, T, T)
                elif attn_weights.dim() == 4:  # [B, H, T, T]
                    pass  # 已经是正确形状
                else:
                    # print(f"WARNING: 意外的注意力权重形状: {attn_weights.shape}")
                    attn_weights = None
                
                if attn_weights is not None:
                    attn_dict = {"A": attn_weights, "Y": attn_out_btc}
                else:
                    attn_dict = {"A": None, "Y": attn_out_btc}
            else:
                # 即使没有注意力权重，也要返回注意力输出用于稳定秩计算
                attn_dict = {"A": None, "Y": attn_out_btc}
        return outputs, attn_dict
