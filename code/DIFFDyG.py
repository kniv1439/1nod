import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import math

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.rotary import apply_rotary_emb
from utils.stable_rank_collector import StableRankCollector


# ===== Metrics helpers: stable rank & spectral/singular norms (no grad) =====
import torch

@torch.no_grad()
def _top_singular_value(X: torch.Tensor, n_iter: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Power iteration to approximate σ_max(X). Works for [T,d] or [T,T]."""
    if X.numel() == 0:
        return torch.tensor(0.0, device=getattr(X, "device", "cpu"))
    # Start with a random vector on the right dimension
    v = torch.randn(X.shape[-1], device=X.device, dtype=X.dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iter):
        # v <- (X^T X) v / ||(X^T X) v||
        v = X.transpose(-1, -2) @ (X @ v)
        v = v / (v.norm() + eps)
    sigma = (X @ v).norm()
    return sigma

@torch.no_grad()
def _stable_rank_xxt(X: torch.Tensor, n_iter: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Stable rank of XX^T: ||X^T X||_F^2 / σ_max(X)^4. X: [T,d] with T>=2."""
    if X.dim() != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        return torch.tensor(1.0, device=X.device if X.is_cuda or X.device else "cpu")
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
        return torch.tensor(0.0, device=A.device)
    J = A.new_ones((T, T)) / float(T)
    M = A - J
    # power iteration on M (equivalently √(λ_max(M^T M)))
    v = torch.randn(T, device=A.device, dtype=A.dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iter):
        v = M.transpose(-1, -2) @ (M @ v)
        v = v / (v.norm() + eps)
    spec = (M @ v).norm()
    return spec


# 尝试导入flash_attn
try:
    from flash_attn import flash_attn_func  # type: ignore
    FLASH_ATTWN_AVAILABLE = True
except Exception:
    FLASH_ATTWN_AVAILABLE = False
    print("WARNING: flash_attn not available, falling back to standard attention")

from utils.rms_norm import RMSNorm


# ============ 内嵌版 多头差分注意力（含 RoPE，支持 FlashAttention 回退） ============

def _lambda_init_fn(depth: int):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class MultiheadFlashDiff2(nn.Module):
    """
    多头差分注意力（DIFF），与原逻辑一致：
    - q,k,v 线性映射 -> 拆成两半 (q1,q2)/(k1,k2)/(v1,v2)
    - 四个分支注意力：11,12,21,22
    - 组合：attn = [11,12] - lambda_full * [21,22]
    - 支持 RoPE；优先用 FlashAttention，无法用时回退为标准注意力
    - 为满足“只在最后一层导出可视化权重”，提供 record_attn=True 时返回 softmax 后的注意力权重：
      A1_hat = softmax(q1 k1^T / sqrt(d)), A2_hat = softmax(q2 k2^T / sqrt(d)), S = A1_hat - lambda_full*A2_hat
    """
    def __init__(self, embed_dim, num_heads, depth, num_kv_heads=None, head_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2 if head_dim is None else head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = _lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        if FLASH_ATTWN_AVAILABLE:
            self.half()

    # 回退标准注意力（支持分块，省内存）
    def _standard_attention(self, q, k, v, causal=True):
        # 输入: [B, T, H, D]
        B, Tq, H, D = q.shape
        Tk = k.shape[1]
        q_ = q.permute(0, 2, 1, 3)  # [B,H,Tq,D]
        k_ = k.permute(0, 2, 1, 3)  # [B,H,Tk,D]
        v_ = v.permute(0, 2, 1, 3)  # [B,H,Tk,D]

        chunk = 64
        out = torch.zeros(B, H, Tq, D, device=q.device, dtype=q.dtype)
        for i in range(0, Tq, chunk):
            j = min(i + chunk, Tq)
            qch = q_[:, :, i:j]  # [B,H,cs,D]
            scores = torch.matmul(qch, k_.transpose(-1, -2)) * self.scaling  # [B,H,cs,Tk]
            if causal:
                causal_mask = torch.triu(torch.ones(j - i, Tk, device=q.device, dtype=torch.bool), diagonal=1 + i - 0)
                scores = scores.masked_fill(causal_mask, float('-inf'))
            w = F.softmax(scores, dim=-1)
            out[:, :, i:j] = torch.matmul(w, v_)
        return out.permute(0, 2, 1, 3)  # [B,Tq,H,D]

    @torch.no_grad()
    def _compute_softmax_weights(self, q, k, causal=True):
        """
        仅用于可视化/导出：标准式 softmax(q k^T / sqrt(d))
        返回 [B, H, Tq, Tk]（按 head 次序）
        """
        B, Tq, H, D = q.shape
        Tk = k.shape[1]
        q_ = q.permute(0, 2, 1, 3).float()  # [B,H,Tq,D]
        k_ = k.permute(0, 2, 1, 3).float()  # [B,H,Tk,D]
        scores = torch.matmul(q_, k_.transpose(-1, -2)) * (1.0 / math.sqrt(D))
        if causal:
            causal_mask = torch.triu(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        w = F.softmax(scores, dim=-1)  # [B,H,Tq,Tk]
        return w

    def forward(self, x, rel_pos, record_attn: bool = False):
        """
        x: [B, T, E]
        rel_pos: (cos, sin) for RoPE
        record_attn: 仅最后一层置 True，用于导出 softmax 权重
        """
        B, T, E = x.size()
        if FLASH_ATTWN_AVAILABLE:
            x = x.half()

        q = self.q_proj(x)  # [B,T,E]
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, 2 * self.head_dim)
        k = k.view(B, T, self.num_kv_heads, 2 * self.head_dim)
        v = v.view(B, T, self.num_kv_heads, 2, self.head_dim)

        device = x.device
        q = q.to(device); k = k.to(device); v = v.to(device)

        cos, sin = rel_pos
        if FLASH_ATTWN_AVAILABLE:
            cos = cos.half(); sin = sin.half()

        # RoPE - 确保在正确的设备上
        if x.is_cuda:
            q = apply_rotary_emb(q, cos, sin, interleaved=True)
            k = apply_rotary_emb(k, cos, sin, interleaved=True)
        else:
            # CPU上跳过RoPE或使用简单实现
            # 暂时跳过RoPE以避免错误
            pass

        # 拆半
        q = q.reshape(B, T, self.num_heads, 2, self.head_dim)
        k = k.reshape(B, T, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]          # [B,T,H,D]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # 四分支注意力
        if FLASH_ATTWN_AVAILABLE:
            attn11 = flash_attn_func(q1, k1, v1, causal=True)
            attn12 = flash_attn_func(q1, k1, v2, causal=True)
            attn21 = flash_attn_func(q2, k2, v1, causal=True)
            attn22 = flash_attn_func(q2, k2, v2, causal=True)
        else:
            attn11 = self._standard_attention(q1, k1, v1, causal=True)
            attn12 = self._standard_attention(q1, k1, v2, causal=True)
            attn21 = self._standard_attention(q2, k2, v1, causal=True)
            attn22 = self._standard_attention(q2, k2, v2, causal=True)

        attn1 = torch.cat([attn11, attn12], dim=-1)  # [B,T,H,2D]
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        out = attn1 - lambda_full * attn2               # [B,T,H,2D]

        out = self.subln(out)
        out = out * (1 - self.lambda_init)
        out = out.reshape(B, T, self.num_heads * 2 * self.head_dim)  # [B,T,E]
        out = self.out_proj(out)

        attn = None
        if record_attn:
            # 导出 A1/A2 以及差分后的 S（均带因果mask；不参与梯度）
            A1 = self._compute_softmax_weights(q1, k1, causal=True)  # [B,H,T,T]
            A2 = self._compute_softmax_weights(q2, k2, causal=True)  # [B,H,T,T]
            # lambda_full: [H] -> [1,H,1,1]
            lam = lambda_full.view(1, -1, 1, 1).to(A1.dtype)
            S = A1 - lam * A2
            attn = {"A1": A1, "A2": A2, "S": S}

        return out, attn
# ========================== END: DIFF 注意力 ==========================


class DIFFDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, num_heads: int, num_layers: int = 28,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu',
                 enable_stable_rank_collection: bool = False, stable_rank_save_dir: str = "results/stable_rank_analysis"):
        super(DIFFDyG, self).__init__()

        self.num_channels = 4  # node, edge, time, neighbor_co_occurrence

        # 确保  num_channels * channel_embedding_dim  可被 2 * num_heads 整除
        if (self.num_channels * channel_embedding_dim) % (2 * num_heads) != 0:
            required_divisor = 2 * num_heads
            remainder = (self.num_channels * channel_embedding_dim) % required_divisor
            if remainder > 0:
                channel_embedding_dim += (required_divisor - remainder) // self.num_channels
                if (self.num_channels * channel_embedding_dim) % required_divisor != 0:
                    channel_embedding_dim += required_divisor // self.num_channels

        # 无 FA 时自动收敛到小配置，节省资源
        if not FLASH_ATTWN_AVAILABLE:
            num_layers = min(num_layers, 6)
            max_input_sequence_length = min(max_input_sequence_length, 128)

        self.channel_embedding_dim = channel_embedding_dim
        self.num_heads = num_heads

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        
        # 稳定秩收集器
        self.enable_stable_rank_collection = enable_stable_rank_collection
        if self.enable_stable_rank_collection:
            self.stable_rank_collector = StableRankCollector(
                save_dir=stable_rank_save_dir, 
                model_name="DIFFDyG"
            )
        else:
            self.stable_rank_collector = None

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # 邻居共现编码器（与原逻辑一致）
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(self.node_feat_dim, self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(self.edge_feat_dim, self.channel_embedding_dim, bias=True),
            'time': nn.Linear(self.time_feat_dim, self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(self.neighbor_co_occurrence_feat_dim, self.channel_embedding_dim, bias=True)
        })

        self.attention_dim = self.num_channels * self.channel_embedding_dim
        if (self.attention_dim // self.num_heads) % 2 != 0:
            raise ValueError(f"attention_dim ({self.attention_dim}) must be divisible by 2*num_heads ({2*self.num_heads}).")

        self.head_dim = self.attention_dim // self.num_heads // 2

        self.transformers = nn.ModuleList([
            TransformerDecoder(
                attention_dim=self.attention_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                depth=layer_idx,
                dropout=self.dropout
            )
            for layer_idx in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=self.node_feat_dim,
            bias=True
        )

    def compute_rotary_embedding(self, seq_len: int, rotary_dim: int):
        max_seq_len = max(seq_len, self.max_input_sequence_length, 512)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=self.device) *
            -(torch.log(torch.tensor(10000.0)) / rotary_dim)
        )
        sinusoid_inp = positions * div_term
        cos = torch.cos(sinusoid_inp)
        sin = torch.sin(sinusoid_inp)
        return cos, sin

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, cos=None, sin=None, rotary_dim=None,
                                                 record_last_attn: bool = False, num_neighbors: int = None):
        # 获取历史邻居 - 使用采样获取更多邻居以增加序列长度
        # 默认采样200个邻居，确保有足够长的序列进行稳定秩分析
        sample_num_neighbors = num_neighbors if num_neighbors is not None else min(200, self.max_input_sequence_length - 1)
        
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=src_node_ids, 
                node_interact_times=node_interact_times,
                num_neighbors=sample_num_neighbors
            )
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=dst_node_ids, 
                node_interact_times=node_interact_times,
                num_neighbors=sample_num_neighbors
            )

        # Padding - 将数组转换为列表格式以兼容pad_sequences
        src_nodes_neighbor_ids_list = [src_nodes_neighbor_ids_list[i] for i in range(len(src_node_ids))]
        src_nodes_edge_ids_list = [src_nodes_edge_ids_list[i] for i in range(len(src_node_ids))]
        src_nodes_neighbor_times_list = [src_nodes_neighbor_times_list[i] for i in range(len(src_node_ids))]
        
        dst_nodes_neighbor_ids_list = [dst_nodes_neighbor_ids_list[i] for i in range(len(dst_node_ids))]
        dst_nodes_edge_ids_list = [dst_nodes_edge_ids_list[i] for i in range(len(dst_node_ids))]
        dst_nodes_neighbor_times_list = [dst_nodes_neighbor_times_list[i] for i in range(len(dst_node_ids))]
        
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, src_padded_node_types = \
            self.pad_sequences(src_node_ids, node_interact_times, src_nodes_neighbor_ids_list,
                               src_nodes_edge_ids_list, src_nodes_neighbor_times_list, self.max_input_sequence_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, dst_padded_node_types = \
            self.pad_sequences(dst_node_ids, node_interact_times, dst_nodes_neighbor_ids_list,
                               dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list, self.max_input_sequence_length)

        # 识别关键节点并更新节点类型标记
        # 对源节点序列：关键节点 = 等于目标节点ID的邻居
        # 对目标节点序列：关键节点 = 等于源节点ID的邻居
        for i in range(len(src_node_ids)):
            # 确保数据类型一致
            dst_node_id = dst_node_ids[i]
            src_node_id = src_node_ids[i]
            
            # 如果tensor，转换为numpy
            if isinstance(dst_node_id, torch.Tensor):
                dst_node_id = dst_node_id.cpu().numpy()
            if isinstance(src_node_id, torch.Tensor):
                src_node_id = src_node_id.cpu().numpy()
            
            # 源节点序列中的关键节点
            src_key_mask = (src_padded_nodes_neighbor_ids[i] == dst_node_id)
            src_padded_node_types[i][src_key_mask] = 1  # 关键节点
            
            # 目标节点序列中的关键节点
            dst_key_mask = (dst_padded_nodes_neighbor_ids[i] == src_node_id)
            dst_padded_node_types[i][dst_key_mask] = 1  # 关键节点

        # 特征对齐
        src_node_raw, src_edge_raw, src_time_feat = \
            self.get_features(node_interact_times, src_padded_nodes_neighbor_ids,
                              src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, self.time_encoder)
        dst_node_raw, dst_edge_raw, dst_time_feat = \
            self.get_features(node_interact_times, dst_padded_nodes_neighbor_ids,
                              dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, self.time_encoder)

        # 邻居共现
        src_nei_co, dst_nei_co = self.neighbor_co_occurrence_encoder(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids
        )

        src_features = torch.cat([
            self.projection_layer['node'](src_node_raw),
            self.projection_layer['edge'](src_edge_raw),
            self.projection_layer['time'](src_time_feat),
            self.projection_layer['neighbor_co_occurrence'](src_nei_co)
        ], dim=-1)
        dst_features = torch.cat([
            self.projection_layer['node'](dst_node_raw),
            self.projection_layer['edge'](dst_edge_raw),
            self.projection_layer['time'](dst_time_feat),
            self.projection_layer['neighbor_co_occurrence'](dst_nei_co)
        ], dim=-1)

        # RoPE 准备
        padded_seq_len = max(src_features.shape[1], dst_features.shape[1])
        embed_dim = src_features.shape[-1]
        headdim_total = embed_dim // self.num_heads
        if rotary_dim is None:
            rotary_dim = min((embed_dim // 2) * 2, headdim_total)
        if cos is None or sin is None:
            cos, sin = self.compute_rotary_embedding(padded_seq_len, rotary_dim)

        # 逐层通过 TransformerDecoder，并收集每层的稳定秩信息
        last_src_attn = last_dst_attn = None
        layer_embeddings_src = []  # 存储每层的src嵌入
        layer_embeddings_dst = []  # 存储每层的dst嵌入
        layer_attentions_src = []  # 存储每层的src注意力
        layer_attentions_dst = []  # 存储每层的dst注意力
        
        # 只在启用稳定秩收集时才收集每层数据
        collect_layer_data = self.enable_stable_rank_collection
        
        for i, transformer in enumerate(self.transformers):
            record = record_last_attn and (i == len(self.transformers) - 1)
            src_features, attn_s = transformer(src_features, cos, sin, rotary_dim, record_attn=record, node_categories=src_padded_node_types if record else None)
            dst_features, attn_d = transformer(dst_features, cos, sin, rotary_dim, record_attn=record, node_categories=dst_padded_node_types if record else None)
            
            # 只在需要时收集每层的嵌入和注意力权重
            if collect_layer_data:
                # 只收集第一个batch的数据，减少内存使用
                layer_embeddings_src.append(src_features[0:1].clone().detach())  # 只保留第一个batch
                layer_embeddings_dst.append(dst_features[0:1].clone().detach())  # 只保留第一个batch
                if attn_s is not None and 'A1' in attn_s:
                    layer_attentions_src.append(attn_s['A1'][0:1].clone().detach())  # 只保留第一个batch
                else:
                    layer_attentions_src.append(None)
                if attn_d is not None and 'A1' in attn_d:
                    layer_attentions_dst.append(attn_d['A1'][0:1].clone().detach())  # 只保留第一个batch
                else:
                    layer_attentions_dst.append(None)
            else:
                # 不收集时，只保留最后一层用于基本功能
                if i == len(self.transformers) - 1:
                    layer_embeddings_src = [src_features[0:1].clone().detach()]
                    layer_embeddings_dst = [dst_features[0:1].clone().detach()]
                    if attn_s is not None and 'A1' in attn_s:
                        layer_attentions_src = [attn_s['A1'][0:1].clone().detach()]
                    else:
                        layer_attentions_src = [None]
                    if attn_d is not None and 'A1' in attn_d:
                        layer_attentions_dst = [attn_d['A1'][0:1].clone().detach()]
                    else:
                        layer_attentions_dst = [None]
            
            if record:
                last_src_attn, last_dst_attn = attn_s, attn_d  # dict 或 None
                
                # 保留注意力矩阵用于分析（不保存到文件）
                pass

        # 聚合
        src_node_embeddings = self.output_layer(torch.mean(src_features, dim=1))
        dst_node_embeddings = self.output_layer(torch.mean(dst_features, dim=1))

        # —— 构造四类节点标记矩阵（与注意力矩阵对齐）——
        node_labels_src = node_labels_dst = None
        if record_last_attn and (last_src_attn is not None):
            # 转换为tensor并扩展到注意力矩阵形状
            src_types = torch.from_numpy(src_padded_node_types).to(self.device)  # [B,T]
            dst_types = torch.from_numpy(dst_padded_node_types).to(self.device)  # [B,T]
            
            # 获取注意力矩阵的实际形状
            if 'A1' in last_src_attn:
                A1_src = last_src_attn['A1']  # [B,H,T,T]
                A1_dst = last_dst_attn['A1'] if last_dst_attn else None
                
                B, H, src_T, _ = A1_src.shape
                if A1_dst is not None:
                    _, _, dst_T, _ = A1_dst.shape
                else:
                    dst_T = src_T
                
                # 创建与注意力矩阵形状完全一致的标签矩阵
                node_labels_src = torch.zeros(B, H, src_T, src_T, dtype=torch.float32, device=self.device)
                node_labels_dst = torch.zeros(B, H, dst_T, dst_T, dtype=torch.float32, device=self.device)
                
                # 将节点标签扩展到注意力矩阵形状
                for b in range(B):
                    for h in range(H):
                        # src标签：按列标记，每列对应一个节点
                        for col in range(src_T):
                            if col < src_types.shape[1]:
                                node_labels_src[b, h, :, col] = src_types[b, col]
                        
                        # dst标签：按列标记，每列对应一个节点
                        for col in range(dst_T):
                            if col < dst_types.shape[1]:
                                node_labels_dst[b, h, :, col] = dst_types[b, col]
            else:
                # 如果没有A1，使用默认的扩展方式
                H = self.num_heads
                node_labels_src = src_types.unsqueeze(1).expand(-1, H, -1, -1).to(torch.float32)  # [B,H,T,T]
                node_labels_dst = dst_types.unsqueeze(1).expand(-1, H, -1, -1).to(torch.float32)  # [B,H,T,T]

        extra = {
            "attn_weights": {  # 单张图（A1）的兼容字段，便于已有可视化脚本
                "src": None if (last_src_attn is None) else last_src_attn.get("A1", None),
                "dst": None if (last_dst_attn is None) else last_dst_attn.get("A1", None),
            },
            "attn_scores": {   # 论文口径：A1/A2/S
                "src": last_src_attn,  # 可能为 None 或 {"A1":...,"A2":...,"S":...}
                "dst": last_dst_attn,
            },
            "node_labels": {"src": node_labels_src, "dst": node_labels_dst},
        }
        
        # 暂时注释掉注意力权重分析
        # if record_last_attn:
        #     attention_analysis = self._analyze_attention_weights(src_padded_node_types, dst_padded_node_types)
        #     if attention_analysis:
        #         extra["attn_scores"].update(attention_analysis)
        
        # ================= 每层稳定秩统计 (参考论文实现) =================
        try:
            with torch.no_grad():
                layer_stable_ranks = {"src": [], "dst": []}
                
                # 收集每一层的token embeddings
                src_embeddings = []
                dst_embeddings = []
                
                # 重新前向传播收集每层embeddings
                src_features_temp = src_features.clone()
                dst_features_temp = dst_features.clone()
                
                for i, transformer in enumerate(self.transformers):
                    # 前向传播但不记录注意力
                    src_features_temp, _ = transformer(src_features_temp, cos, sin, rotary_dim, record_attn=False)
                    dst_features_temp, _ = transformer(dst_features_temp, cos, sin, rotary_dim, record_attn=False)
                    
                    # 收集当前层的embeddings
                    src_embeddings.append(src_features_temp.clone())
                    dst_embeddings.append(dst_features_temp.clone())
                
                # 计算每层的稳定秩
                for l, (src_emb, dst_emb) in enumerate(zip(src_embeddings, dst_embeddings)):
                    # SRC
                    src_stable_ranks = []
                    for b in range(src_emb.shape[0]):
                        vm = torch.from_numpy(src_padded_node_types[b] != 4).to(self.device)
                        if vm.sum() >= 2:  # 至少需要2个有效token
                            X = src_emb[b, vm, :]  # [T_valid, E]
                            try:
                                sigma = X @ X.T  # [T_valid, T_valid]
                                s = torch.linalg.svdvals(sigma)
                                stable_rank = ((s**2).sum() / (s**2).max()).item()
                                src_stable_ranks.append(stable_rank)
                            except:
                                src_stable_ranks.append(float('nan'))
                    
                    # DST
                    dst_stable_ranks = []
                    for b in range(dst_emb.shape[0]):
                        vm = torch.from_numpy(dst_padded_node_types[b] != 4).to(self.device)
                        if vm.sum() >= 2:  # 至少需要2个有效token
                            X = dst_emb[b, vm, :]  # [T_valid, E]
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
                B = src_features.shape[0]
                
                # 只在启用稳定秩收集且有必要数据时才计算
                if (B > 0 and layer_embeddings_src and layer_embeddings_dst and 
                    self.enable_stable_rank_collection and 
                    len(layer_embeddings_src) > 0):
                    
                    # 只对最后一层计算稳定秩，大幅减少计算量
                    last_layer_idx = len(layer_embeddings_src) - 1
                    src_emb = layer_embeddings_src[last_layer_idx]
                    dst_emb = layer_embeddings_dst[last_layer_idx]
                    
                    # 计算src的稳定秩（使用与stable_rank_collector.py完全相同的方法）
                    if src_emb.shape[1] >= 2:  # 至少需要2个token
                        try:
                            # 取第一个batch样本
                            X = src_emb[0]  # [T, d]
                            
                            # 使用stable_rank_collector.py的compute_stable_rank方法（xxt方法）
                            # XX^T 方法：||X^T X||_F^2 / σ_max(X)^4
                            G = X.transpose(-1, -2) @ X  # [d, d]
                            fro_norm_sq = (G * G).sum()
                            
                            # 计算最大奇异值
                            _, s, _ = torch.svd(X)
                            sigma_max = s.max()
                            
                            stable_rank = (fro_norm_sq / (sigma_max ** 4 + 1e-8)).clamp_min(1.0).item()
                            
                            # 计算谱间隙（使用与stable_rank_collector.py相同的方法）
                            gap = None
                            s_op = None
                            if last_src_attn is not None:
                                if ("A1" in last_src_attn) and (last_src_attn["A1"] is not None):
                                    A = last_src_attn["A1"][0].mean(dim=0)   # [T, T] 头均值
                                    # 使用stable_rank_collector.py的compute_spectral_gap方法
                                    T = A.shape[0]
                                    if T > 1:
                                        J = torch.ones_like(A) / float(T)
                                        M = A - J
                                        _, s_gap, _ = torch.svd(M)
                                        gap = float(s_gap.max())
                                
                                if ("S" in last_src_attn) and (last_src_attn["S"] is not None):
                                    S = last_src_attn["S"][0].mean(dim=0)     # [T, T] 头均值（差分矩阵）
                                    _, s_diff, _ = torch.svd(S)
                                    s_op = float(s_diff.max())
                            
                            metrics["src"].append({
                                "layer": last_layer_idx,
                                "T_valid": src_emb.shape[1],
                                "stable_rank": stable_rank,
                                "gap_norm": gap,
                                "S_op_norm": s_op,
                            })
                        except Exception as e:
                            print(f"Warning: SRC stable rank calculation failed: {e}")
                    
                    # 计算dst的稳定秩（使用与stable_rank_collector.py完全相同的方法）
                    if dst_emb.shape[1] >= 2:
                        try:
                            # 取第一个batch样本
                            X = dst_emb[0]  # [T, d]
                            
                            # 使用stable_rank_collector.py的compute_stable_rank方法（xxt方法）
                            # XX^T 方法：||X^T X||_F^2 / σ_max(X)^4
                            G = X.transpose(-1, -2) @ X  # [d, d]
                            fro_norm_sq = (G * G).sum()
                            
                            # 计算最大奇异值
                            _, s, _ = torch.svd(X)
                            sigma_max = s.max()
                            
                            stable_rank = (fro_norm_sq / (sigma_max ** 4 + 1e-8)).clamp_min(1.0).item()
                            
                            # 计算谱间隙（使用与stable_rank_collector.py相同的方法）
                            gap = None
                            s_op = None
                            if last_dst_attn is not None:
                                if ("A1" in last_dst_attn) and (last_dst_attn["A1"] is not None):
                                    A = last_dst_attn["A1"][0].mean(dim=0)   # [T, T] 头均值
                                    # 使用stable_rank_collector.py的compute_spectral_gap方法
                                    T = A.shape[0]
                                    if T > 1:
                                        J = torch.ones_like(A) / float(T)
                                        M = A - J
                                        _, s_gap, _ = torch.svd(M)
                                        gap = float(s_gap.max())
                                
                                if ("S" in last_dst_attn) and (last_dst_attn["S"] is not None):
                                    S = last_dst_attn["S"][0].mean(dim=0)     # [T, T] 头均值（差分矩阵）
                                    _, s_diff, _ = torch.svd(S)
                                    s_op = float(s_diff.max())
                            
                            metrics["dst"].append({
                                "layer": last_layer_idx,
                                "T_valid": dst_emb.shape[1],
                                "stable_rank": stable_rank,
                                "gap_norm": gap,
                                "S_op_norm": s_op,
                            })
                        except Exception as e:
                            print(f"Warning: DST stable rank calculation failed: {e}")
                
                extra["metrics"] = metrics
        except Exception as _e:
            extra["metrics_error"] = str(_e)
        
        # ================= 新增：使用StableRankCollector进行稳定秩收集 =================
        if self.enable_stable_rank_collection and layer_embeddings_src and self.stable_rank_collector is not None:
            try:
                # 只收集最后一层的稳定秩指标以提高性能
                last_layer_idx = len(layer_embeddings_src) - 1
                
                # SRC层级指标
                if last_layer_idx < len(layer_embeddings_src):
                    src_metrics = self.stable_rank_collector.collect_batch_metrics(
                        embeddings=layer_embeddings_src[last_layer_idx],
                        attention_weights=layer_attentions_src[last_layer_idx] if last_layer_idx < len(layer_attentions_src) else None,
                        layer_idx=last_layer_idx,
                        batch_idx=getattr(self, '_batch_idx', 0),
                        node_types=src_padded_node_types
                    )
                    src_metrics['data_type'] = 'src'
                    self.stable_rank_collector.add_layer_data([src_metrics])
                
                # DST层级指标  
                if last_layer_idx < len(layer_embeddings_dst):
                    dst_metrics = self.stable_rank_collector.collect_batch_metrics(
                        embeddings=layer_embeddings_dst[last_layer_idx],
                        attention_weights=layer_attentions_dst[last_layer_idx] if last_layer_idx < len(layer_attentions_dst) else None,
                        layer_idx=last_layer_idx,
                        batch_idx=getattr(self, '_batch_idx', 0),
                        node_types=dst_padded_node_types
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
        elif self.enable_stable_rank_collection:
            # 调试信息：为什么没有收集到数据
            debug_info = {
                "has_collector": self.stable_rank_collector is not None,
                "has_src_embeddings": bool(layer_embeddings_src),
                "has_dst_embeddings": bool(layer_embeddings_dst),
                "num_src_embeddings": len(layer_embeddings_src) if layer_embeddings_src else 0,
                "num_dst_embeddings": len(layer_embeddings_dst) if layer_embeddings_dst else 0,
                "enabled": self.enable_stable_rank_collection
            }
            extra["stable_rank_debug"] = debug_info

        return src_node_embeddings, dst_node_embeddings, extra

    # 暂时注释掉注意力权重分析方法
    # def _analyze_attention_weights(self, src_categories, dst_categories):
    #     """
    #     分析注意力权重，统计四类节点的注意力分配占比
    #     """
    #     extra_info = {}
    #     
    #     # 获取最后一层的注意力权重
    #     for i, transformer in enumerate(self.transformers):
    #         if hasattr(transformer, 'last_attn_scores') and transformer.last_attn_scores is not None:
    #             attn_scores = transformer.last_attn_scores  # 这是差分注意力的正值部分 S+
    #             node_categories = transformer.node_categories
    #                 
    #             if attn_scores is not None and node_categories is not None:
    #                 # 处理源节点和目标节点的注意力权重
    #                 for side, categories in [('src', src_categories), ('dst', dst_categories)]:
    #                     if categories is not None:
    #                         # 确保categories的形状与attn_scores匹配
    #                         B, H, T, _ = attn_scores.shape
    #                         if categories.shape[1] != T:
    #                             # 调整categories的形状以匹配注意力矩阵
    #                             categories = categories[:, :T] if categories.shape[1] > T else categories
    #                             
    #                         # 初始化类别权重
    #                         category_weights = torch.zeros(4, device=attn_scores.device)  # [self, key, other, pad]
    #                             
    #                         # 对所有head和query求和
    #                         total_attn = torch.sum(attn_scores, dim=(1, 2))  # [B, T]
    #                             
    #                         # 按类别统计注意力权重
    #                         for b in range(B):
    #                             for key_idx in range(min(categories.shape[1], total_attn.shape[1])):
    #                                     if key_idx < categories.shape[1]:
    #                                         key_cat = int(categories[b, key_idx])
    #                                         
    #                                         # 正确的类别映射：
    #                                         # 0=self, 1=key, 2=secondary_key, 3=other, 4=pad
    #                                         if key_cat == 0:  # self token
    #                                         mapped_cat = 0
    #                                     elif key_cat == 1:  # key node
    #                                         mapped_cat = 1
    #                                     elif key_cat == 2:  # secondary key (也是关键节点)
    #                                         mapped_cat = 1
    #                                     elif key_cat == 3:  # other node
    #                                         mapped_cat = 2
    #                                     elif key_cat == 4:  # padding
    #                                         mapped_cat = 3
    #                                     else:  # 其他情况，默认为other
    #                                         mapped_cat = 2
    #                                         
    #                                     if key_idx < total_attn.shape[1]:
    #                                         category_weights[mapped_cat] += total_attn[b, key_idx].item()
    #                             
    #                         # 归一化
    #                         total_weight = category_weights.sum()
    #                         if total_weight > 0:
    #                             category_weights = category_weights / total_weight
    #                             
    #                         # 构造返回的数据结构
    #                         share_dict = {
    #                             'self': category_weights[0].item(),
    #                             'key': category_weights[1].item(),
    #                             'other': category_weights[2].item(),
    #                             'pad': category_weights[3].item()
    #                         }
    #                         
    #                         if side not in extra_info:
    #                             extra_info[side] = {}
    #                         extra_info[side]['S'] = {'share': share_dict}
    #                             
    #                 break  # 只处理最后一层
    #     
    #     return extra_info

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list,
                      nodes_edge_ids_list: list, nodes_neighbor_times_list: list, max_input_sequence_length: int = 256):
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            current_len = len(nodes_neighbor_ids_list[idx])
            if current_len > max_seq_length:
                max_seq_length = current_len
        max_seq_length += 1
        max_seq_length = ((max_seq_length + self.num_heads - 1) // self.num_heads) * self.num_heads
        max_seq_length = min(max_seq_length, self.max_input_sequence_length)
        if not FLASH_ATTWN_AVAILABLE and max_seq_length > 64:
            max_seq_length = ((64 + self.num_heads - 1) // self.num_heads) * self.num_heads

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length), dtype=np.float32)
        # 新增：节点类型标记 (0=self, 1=关键, 2=次要关键, 3=无关, 4=padding)
        padded_node_types = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        
        for idx in range(len(node_ids)):
            # 位置0：self token
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
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
                # 先全部标记为无关节点
                padded_node_types[idx, 1:1 + neighbor_len] = 3  # 无关节点
                
                # 计算次要关键节点数量（最近的20%）
                secondary_key_count = max(1, int(neighbor_len * 0.1))
                # 最近的secondary_key_count个节点标记为次要关键节点
                padded_node_types[idx, 1:1 + secondary_key_count] = 2  # 次要关键节点
                
                # 剩余位置标记为padding
                padded_node_types[idx, 1 + neighbor_len:] = 4  # padding
        
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times, padded_node_types

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray, padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.cpu().numpy()
        if isinstance(padded_nodes_neighbor_ids, torch.Tensor):
            padded_nodes_neighbor_ids = padded_nodes_neighbor_ids.cpu().numpy()
        if isinstance(padded_nodes_edge_ids, torch.Tensor):
            padded_nodes_edge_ids = padded_nodes_edge_ids.cpu().numpy()
        if isinstance(padded_nodes_neighbor_times, torch.Tensor):
            padded_nodes_neighbor_times = padded_nodes_neighbor_times.cpu().numpy()

        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device)
        )
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        padded_nodes_neighbor_node_raw_features = padded_nodes_neighbor_node_raw_features.to(self.device)
        padded_nodes_edge_raw_features = padded_nodes_edge_raw_features.to(self.device)
        padded_nodes_neighbor_time_features = padded_nodes_neighbor_time_features.to(self.device)

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

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
                model_name="DIFFDyG"
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

    def forward(self, batch):
        src_node_ids = batch['src_node_ids']
        dst_node_ids = batch['dst_node_ids']
        node_interact_times = batch['node_interact_times']
        src_emb, dst_emb, extra = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times, record_last_attn=True
        )
        return src_emb, dst_emb, extra


class TransformerDecoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, head_dim: int, depth: int, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mha = MultiheadFlashDiff2(
            embed_dim=attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([
            nn.Linear(attention_dim, 3 * attention_dim),
            nn.Linear((3 * attention_dim) // 2, attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            RMSNorm(attention_dim),
            RMSNorm(attention_dim)
        ])

    def forward(self, src_features, cos, sin, rotary_dim, record_attn: bool = False, node_categories=None):
        if FLASH_ATTWN_AVAILABLE:
            src_features = src_features.half()

        x = src_features.transpose(0, 1)  # [T,B,E]
        x = self.norm_layers[0](x)
        T, B, E = x.shape
        assert E == self.num_heads * 2 * self.head_dim, f"E={E} vs {self.num_heads*2*self.head_dim}"

        # 截取 RoPE
        if cos.shape[0] < T:
            raise ValueError(f"cos/sin length ({cos.shape[0]}) < seq_len ({T})")
        cos_t = cos[:T, :rotary_dim // 2]
        sin_t = sin[:T, :rotary_dim // 2]

        # MHA (DIFF)
        h, attn_dict = self.mha(
            x.transpose(0, 1),  # [B,T,E]
            rel_pos=(cos_t, sin_t),
            record_attn=record_attn
        )  # h: [B,T,E]; attn_dict: None 或 {"A1":...,"A2":...,"S":...}

        # 保存注意力分数和类别信息以供分析
        if record_attn and attn_dict is not None and 'S' in attn_dict:
            # 取差分注意力的正值部分
            S = attn_dict['S']  # [B, H, T, T]
            self.last_attn_scores = torch.clamp(S, min=0.0)  # S+ = max(S, 0)
            self.node_categories = node_categories

        # 维度对齐：h 为 [B,T,E]，与 src_features 一致，直接残差相加
        outputs = src_features + self.dropout(h)

        z = self.linear_layers[0](self.norm_layers[1](outputs))
        x1, x2 = torch.chunk(z, chunks=2, dim=-1)
        swiglu = F.gelu(x1) * torch.sigmoid(x2)
        z2 = self.linear_layers[1](self.dropout(swiglu))
        outputs = outputs + self.dropout(z2)
        return outputs, attn_dict


class NeighborCooccurrenceEncoder(nn.Module):
    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(1, self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(self.neighbor_co_occurrence_feat_dim, self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        for src_ids, dst_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):
            src_unique, src_inv, src_counts = np.unique(src_ids, return_inverse=True, return_counts=True)
            src_counts_expand = torch.from_numpy(src_counts[src_inv]).float().to(self.device)
            src_map = dict(zip(src_unique, src_counts))

            dst_unique, dst_inv, dst_counts = np.unique(dst_ids, return_inverse=True, return_counts=True)
            dst_counts_expand = torch.from_numpy(dst_counts[dst_inv]).float().to(self.device)
            dst_map = dict(zip(dst_unique, dst_counts))

            src_in_dst = torch.from_numpy(src_ids.copy()).apply_(lambda nid: dst_map.get(nid, 0.0)).float().to(self.device)
            dst_in_src = torch.from_numpy(dst_ids.copy()).apply_(lambda nid: src_map.get(nid, 0.0)).float().to(self.device)

            src_padded_nodes_appearances.append(torch.stack([src_counts_expand, src_in_dst], dim=1))
            dst_padded_nodes_appearances.append(torch.stack([dst_in_src, dst_counts_expand], dim=1))

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0
        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_app, dst_app = self.count_nodes_appearances(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids)
        src_feat = self.neighbor_co_occurrence_encode_layer(src_app.unsqueeze(-1)).sum(dim=2)
        dst_feat = self.neighbor_co_occurrence_encode_layer(dst_app.unsqueeze(-1)).sum(dim=2)
        return src_feat, dst_feat

