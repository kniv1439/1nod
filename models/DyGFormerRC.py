import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from typing import Union, Optional
from einops import rearrange

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.rotary import apply_rotary_emb
from utils.RoPE import LlamaDynamicNTKScalingRotaryEmbedding


class AttenionDyG(nn.Module):
    """
    适配动态图的 CARD 注意力类：
    - 使用 LayerNorm 替代 BatchNorm
    - 支持动态扩展的 EMA 矩阵
    - 适配 4D 输入 (B, nvars, H, C)
    """
    def __init__(self, config, over_hidden=False, *args, **kwargs):
        super().__init__()
        
        self.over_hidden = over_hidden
        self.n_heads = config.n_heads
        self.c_in = config.enc_in  # num_channels
        self.d_model = config.d_model  # channel_embedding_dim
        self.qkv = nn.Linear(self.d_model, self.d_model * 3, bias=True)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = self.d_model // self.n_heads
        # 注意：head_dim 的实际值在 forward 中动态计算，这里只存储理论值
        
        # ✅ 使用 LayerNorm 替代 BatchNorm
        self.norm_post1 = nn.LayerNorm(self.d_model)
        self.norm_post2 = nn.LayerNorm(self.d_model)
        self.norm_attn = nn.LayerNorm(self.d_model)
        
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        
        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, self.d_model, bias=True)
        )
        
        self.ff_2 = nn.Sequential(
            nn.Linear(self.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, self.d_model, bias=True)
        )
        
        self.merge_size = config.merge_size
        self.alpha = config.alpha
        
        # ✅ 初始化一个较大的 EMA 矩阵，支持动态扩展
        initial_ema_size = max(config.enc_in, 512, config.dp_rank)  # 512 作为初始大小
        ema_matrix = torch.zeros((initial_ema_size, initial_ema_size))
        ema_matrix[0][0] = 1
        for i in range(1, initial_ema_size):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i-1][j] * (1 - self.alpha)
            ema_matrix[i][i] = self.alpha
        self.register_buffer('ema_matrix', ema_matrix)
       
    def ema(self, src):
        """
        ✅ 动态扩展 EMA 矩阵
        src: (B, nvars, n_heads, H, head_dim)
        """
        # 获取序列长度 H（在倒数第二个维度）
        H = src.shape[-2]
        
        # 动态扩展 EMA 矩阵
        if H > self.ema_matrix.shape[0]:
            device = self.ema_matrix.device
            old_size = self.ema_matrix.shape[0]
            new_ema = torch.zeros((H, H), device=device)
            new_ema[:old_size, :old_size] = self.ema_matrix.data
            for i in range(old_size, H):
                for j in range(i):
                    new_ema[i, j] = new_ema[i-1, j] * (1 - self.alpha)
                new_ema[i, i] = self.alpha
            self.ema_matrix = nn.Parameter(new_ema, requires_grad=False)
        
        return torch.einsum('bnhad,ga->bnhgd', src, self.ema_matrix[:H, :H])

    def dynamic_projection(self, src, mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum('bnhef,bnhec->bnhcf', src, src_dp)
        return src_dp
        
    def forward(self, src, cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None, *args, **kwargs):
        """
        输入: src shape (B, nvars, H, C)
        B: batch_size
        nvars: num_channels
        H: num_patches (序列长度)
        C: channel_embedding_dim
        
        可选参数:
        cos: Tensor, shape (H, rotary_dim//2) 或 (H, head_dim//2)，用于 RoPE
        sin: Tensor, shape (H, rotary_dim//2) 或 (H, head_dim//2)，用于 RoPE
        """
        B, nvars, H, C = src.shape
        
        # QKV 投影
        # 注意：由于 head_dim 可能被调整为偶数，需要确保维度匹配
        actual_head_dim = C // self.n_heads
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.n_heads, actual_head_dim).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nvars, n_heads, H, actual_head_dim)
        
        # ✅ 应用 RoPE（如果提供了 cos 和 sin）
        use_rope = (cos is not None) and (sin is not None)
        if use_rope:
            # 确保 head_dim 是偶数且与 RoPE 维度匹配
            rotary_dim = cos.shape[1] * 2  # cos/sin 的形状是 (seq_len, rotary_dim//2)
            
            # 确保 rotary_dim 不超过 actual_head_dim，且是偶数
            rotary_dim = min(rotary_dim, actual_head_dim)
            if rotary_dim % 2 != 0:
                rotary_dim = rotary_dim - 1
            
            # 只对前 rotary_dim 个维度应用 RoPE
            if rotary_dim > 0:
                # 重塑 Q 和 K 以应用 RoPE: (B, nvars, n_heads, H, actual_head_dim) -> (B*nvars, H, n_heads, rotary_dim)
                q_rope = q[..., :rotary_dim].reshape(B * nvars, H, self.n_heads, rotary_dim)
                k_rope = k[..., :rotary_dim].reshape(B * nvars, H, self.n_heads, rotary_dim)
                
                # 应用 RoPE（apply_rotary_emb 期望 (batch, seq_len, n_heads, head_dim)）
                # 注意：cos 和 sin 的形状是 (seq_len, rotary_dim//2)，需要确保匹配
                cos_rope = cos[:, :rotary_dim//2] if cos.shape[1] >= rotary_dim//2 else cos
                sin_rope = sin[:, :rotary_dim//2] if sin.shape[1] >= rotary_dim//2 else sin
                
                q_rope = apply_rotary_emb(q_rope, cos_rope, sin_rope, interleaved=True)
                k_rope = apply_rotary_emb(k_rope, cos_rope, sin_rope, interleaved=True)
                
                # 重塑回原形状并更新 q, k
                q_rope = q_rope.reshape(B, nvars, self.n_heads, H, rotary_dim)
                k_rope = k_rope.reshape(B, nvars, self.n_heads, H, rotary_dim)
                
                # 如果 actual_head_dim > rotary_dim，保留剩余维度
                if actual_head_dim > rotary_dim:
                    q = torch.cat([q_rope, q[..., rotary_dim:]], dim=-1)
                    k = torch.cat([k_rope, k[..., rotary_dim:]], dim=-1)
                else:
                    q = q_rope
                    k = k_rope
    
        # Token 维度注意力（在 H 维度上）
        if not self.over_hidden: 
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k)) / (actual_head_dim ** 0.5)
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)
        else:
            # Channel 注意力：使用动态投影
            v_dp, k_dp = self.dynamic_projection(v, self.dp_v), self.dynamic_projection(k, self.dp_k)
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp)) / (actual_head_dim ** 0.5)
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)
        
        # Hidden/Channel 维度注意力（在 nvars 维度上）
        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q, k) / (q.shape[-2] ** 0.5)
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1))
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)

        # ✅ 简化 rearrange 操作：如果 merge_size=1，直接 reshape
        merge_size = self.merge_size
        if merge_size == 1:
            # 简化版本：直接 reshape（使用 actual_head_dim 而不是 self.head_dim）
            output1 = output_along_token.reshape(B * nvars, H, self.n_heads * actual_head_dim)
            output2 = output_along_hidden.reshape(B * nvars, H, self.n_heads * actual_head_dim)
        else:
            # 使用 rearrange（仅在 merge_size > 1 时）
            output1 = rearrange(
                output_along_token.reshape(B * nvars, -1, actual_head_dim),
                'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d',
                hl1=self.n_heads // merge_size,
                hl2=output_along_token.shape[-2],
                hl3=merge_size
            ).reshape(B * nvars, -1, self.n_heads * actual_head_dim)
            
            output2 = rearrange(
                output_along_hidden.reshape(B * nvars, -1, actual_head_dim),
                'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d',
                hl1=self.n_heads // merge_size,
                hl2=output_along_hidden.shape[-2],
                hl3=merge_size
            ).reshape(B * nvars, -1, self.n_heads * actual_head_dim)

        # ✅ 使用 LayerNorm（无需 Transpose）
        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B, nvars, H, self.n_heads * actual_head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B, nvars, H, self.n_heads * actual_head_dim)

        # 融合两个注意力的输出
        src2 = self.ff_1(output1) + self.ff_2(output2)
        
        # 残差连接
        src = src + src2
        src = src.reshape(B * nvars, H, self.n_heads * actual_head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B, nvars, H, self.n_heads * actual_head_dim)
        return src


class DyGFormerRC(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu',
                 dp_rank: int = 8, alpha: float = 0.9, d_ff: int = None, merge_size: int = 1, momentum: float = 0.1):
        """
        OCARDDyG: Optimized CARD architecture for dynamic graphs.
        
        Improvements over original CARD:
        - Uses LayerNorm instead of BatchNorm (better for variable-length sequences)
        - Dynamic EMA matrix expansion (adapts to variable patch numbers)
        - Simplified dimension handling (reduced permute operations)
        - Maintains dual sequential attention architecture (Channel -> Token)
        
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
        :param dp_rank: int, dynamic projection rank
        :param alpha: float, EMA decay factor
        :param d_ff: int, feed-forward dimension (default: 4 * channel_embedding_dim)
        :param merge_size: int, head merge size for rearrange operation
        :param momentum: float, momentum for normalization (not used with LayerNorm)
        """
        super(DyGFormerRC, self).__init__()

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
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4

        # ✅ 计算 RoPE 相关维度
        # 确保 head_dim 是偶数（RoPE 要求）
        head_dim = self.channel_embedding_dim // self.num_heads
        if head_dim % 2 != 0:
            head_dim = head_dim - 1
        self.head_dim = head_dim
        self.rotary_dim = head_dim  # rotary_dim 通常等于 head_dim
        
        # 动态NTK缩放的RoPE：缓存不同rotary_dim的实例
        self._rotary_emb_cache = {}

        # 创建配置对象用于 CARD 注意力
        d_ff = d_ff if d_ff is not None else 4 * self.channel_embedding_dim
        # ✅ 不再需要固定的 max_patches，EMA 矩阵会动态扩展
        self.card_config = type('Config', (), {
            'd_model': self.channel_embedding_dim,
            'n_heads': self.num_heads,
            'enc_in': self.num_channels,
            'total_token_number': max_input_sequence_length // patch_size,  # 仅用于初始化，实际会动态扩展
            'dp_rank': dp_rank,
            'alpha': alpha,
            'd_ff': d_ff,
            'merge_size': merge_size,
            'dropout': self.dropout,
            'momentum': momentum
        })()

        # ✅ 使用改进的 AttenionDyG 类（LayerNorm + 动态 EMA）
        # Token注意力：在patch维度（时间维度）上
        self.Attentions_over_token = nn.ModuleList([
            AttenionDyG(self.card_config, over_hidden=False) 
            for _ in range(self.num_layers)
        ])
        # Channel注意力：在channel维度上
        self.Attentions_over_channel = nn.ModuleList([
            AttenionDyG(self.card_config, over_hidden=True) 
            for _ in range(self.num_layers)
        ])
        # MLP和归一化层
        self.Attentions_mlp = nn.ModuleList([
            nn.Linear(self.channel_embedding_dim, self.channel_embedding_dim) 
            for _ in range(self.num_layers)
        ])
        self.Attentions_dropout = nn.ModuleList([
            nn.Dropout(self.dropout) 
            for _ in range(self.num_layers)
        ])
        # ✅ 使用 LayerNorm 替代 BatchNorm
        self.Attentions_norm = nn.ModuleList([
            nn.LayerNorm(self.channel_embedding_dim)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

        # 位置编码已移除
        # self.W_pos_embed = nn.Parameter(torch.randn(self.max_input_sequence_length // self.patch_size, self.num_channels * self.channel_embedding_dim) * 1e-2)

    # --------------------- RoPE（动态NTK缩放） ---------------------
    def _get_rotary_emb(self, seq_len: int, rotary_dim: int):
        """根据rotary_dim获取或创建LlamaDynamicNTKScalingRotaryEmbedding实例"""
        key = rotary_dim
        if key not in self._rotary_emb_cache:
            self._rotary_emb_cache[key] = LlamaDynamicNTKScalingRotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=max(seq_len, self.max_input_sequence_length),
                base=10000,
                device=self.device,
                scaling_factor=1.0,
            ).to(self.device)
        return self._rotary_emb_cache[key]

    def compute_rotary_embedding(self, seq_len: int, rotary_dim: int):
        """使用LlamaDynamicNTKScalingRotaryEmbedding生成(cos, sin)。返回形状(seq_len, rotary_dim//2)。"""
        rotary_emb = self._get_rotary_emb(seq_len, rotary_dim)
        # 创建一个虚拟张量用于推理dtype & device
        # 确保 dummy 的 seq_len 维度足够大，以触发 rotary_emb 更新缓存（如果需要）
        dummy = torch.zeros(1, 1, seq_len, rotary_dim, device=self.device, dtype=torch.float32)
        cos_cached, sin_cached = rotary_emb(dummy, seq_len=seq_len)
        # squeeze掉前两维 -> (seq_len, rotary_dim)
        cos_full = cos_cached.squeeze(0).squeeze(0)  # (seq_len, rotary_dim)
        sin_full = sin_cached.squeeze(0).squeeze(0)
        # 确保返回的长度正确
        if cos_full.shape[0] < seq_len:
            # 如果长度不够，说明 rotary_emb 的缓存没有正确更新，需要重新计算
            # 这种情况不应该发生，但如果发生了，通过重复最后一个元素来扩展
            last_cos = cos_full[-1:]  # (1, rotary_dim)
            last_sin = sin_full[-1:]  # (1, rotary_dim)
            repeat_times = seq_len - cos_full.shape[0]
            cos_full = torch.cat([cos_full, last_cos.repeat(repeat_times, 1)], dim=0)
            sin_full = torch.cat([sin_full, last_sin.repeat(repeat_times, 1)], dim=0)
        # 取前一半维度，形状符合apply_rotary_emb要求
        cos = cos_full[:seq_len, : rotary_dim // 2].contiguous()
        sin = sin_full[:seq_len, : rotary_dim // 2].contiguous()
        return cos, sin

    def forward(self, batch):
        """
        前向传播方法，接受一个 batch 字典，返回源节点和目标节点的嵌入。
        """
        src_node_ids = batch['src_node_ids']
        dst_node_ids = batch['dst_node_ids']
        node_interact_times = batch['node_interact_times']

        # 调用 compute_src_dst_node_temporal_embeddings 方法计算嵌入
        src_node_embeddings, dst_node_embeddings = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times
        )

        return src_node_embeddings, dst_node_embeddings

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: Union[np.ndarray, torch.Tensor], 
                                                 dst_node_ids: Union[np.ndarray, torch.Tensor], 
                                                 node_interact_times: Union[np.ndarray, torch.Tensor],
                                                 num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray or Tensor, shape (batch_size, )
        :param dst_node_ids: ndarray or Tensor, shape (batch_size, )
        :param node_interact_times: ndarray or Tensor, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample (not used in CARDDyG, kept for interface compatibility)
        :return:
        """
        # 如果输入是 numpy 数组，则转换为 Tensor
        if isinstance(src_node_ids, np.ndarray):
            src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        if isinstance(dst_node_ids, np.ndarray):
            dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        if isinstance(node_interact_times, np.ndarray):
            node_interact_times = torch.from_numpy(node_interact_times).to(self.device)

        # 确保输入已经在正确的设备上
        src_node_ids = src_node_ids.to(self.device)
        dst_node_ids = dst_node_ids.to(self.device)
        node_interact_times = node_interact_times.to(self.device)

        # 在调用 neighbor_sampler 之前，将张量移动到 CPU 并转换为 numpy 数组
        src_node_ids_cpu = src_node_ids.cpu().numpy()
        node_interact_times_cpu = node_interact_times.cpu().numpy()

        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids_cpu, node_interact_times=node_interact_times_cpu)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids.cpu().numpy(), node_interact_times=node_interact_times_cpu)

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

        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # align the patch encoding dimension
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
        # 保持4D形状，不展平！这是关键修改
        # patches_data: (B, L, num_channels, channel_embedding_dim)
        # 其中 B=batch_size, L=num_patches, num_channels=4, channel_embedding_dim=d_model

        # 应用 CARD 的双重注意力机制
        # 按照 CARDformer 的设计：先channel注意力，再token注意力
        b, l, c, h = patches_data.shape  # B, L, num_channels, channel_embedding_dim
        
        # ✅ 生成 RoPE 的 cos 和 sin（用于 Token 注意力）
        # Token 注意力在 patch 维度（L）上，所以序列长度是 l
        cos, sin = self.compute_rotary_embedding(seq_len=l, rotary_dim=self.rotary_dim)
        
        for a_token, a_channel, mlp, drop, norm in zip(
            self.Attentions_over_token, 
            self.Attentions_over_channel,
            self.Attentions_mlp,
            self.Attentions_dropout,
            self.Attentions_norm
        ):
            # ✅ 简化：只需要一次 permute
            # Channel注意力：输入需要 (B, nvars, H, C) = (B, num_channels, L, channel_embedding_dim)
            patches_4d = patches_data.permute(0, 2, 1, 3)  # (B, L, C, H) -> (B, C, L, H)
            output_channel = a_channel(patches_4d).permute(0, 2, 1, 3)  # 转回 (B, L, C, H)
            
            # Token注意力：再次 permute，并传入 RoPE 的 cos 和 sin
            output_channel_4d = output_channel.permute(0, 2, 1, 3)  # (B, L, C, H) -> (B, C, L, H)
            output_token = a_token(output_channel_4d, cos=cos, sin=sin).permute(0, 2, 1, 3)  # 转回 (B, L, C, H)
            
            # ✅ 融合：简化处理
            output_sum = (output_channel + output_token).reshape(b * l * c, h)  # (B*L*C, d_model)
            output_mlp = mlp(output_sum).reshape(b, l, c, h)  # (B, L, C, d_model)
            outputs = drop(output_mlp) + patches_data
            
            # ✅ 归一化：使用 LayerNorm，简化处理
            # 对每个位置的特征进行归一化: (B*L*C, d_model)
            outputs_flat = outputs.reshape(b * l * c, h)
            outputs_normed = norm(outputs_flat).reshape(b, l, c, h)
            patches_data = outputs_normed

        # 展平channel维度用于输出层
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

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

    def get_features(self, node_interact_times, padded_nodes_neighbor_ids, padded_nodes_edge_ids,
                     padded_nodes_neighbor_times, time_encoder):
        """
        get node, edge and time features
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

        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids).to(self.device)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids).to(self.device)]
        # 计算时间差
        time_diffs = node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
        # 转成 tensor 并放到 device
        time_diffs_tensor = torch.from_numpy(time_diffs).float().to(self.device)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=time_diffs_tensor)

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0).to(self.device)] = 0.0

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

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cuda:0'):
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


class CARDTokenAttentionEncoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, dp_rank: int, alpha: float = 0.9, dropout: float = 0.1):
        super().__init__()
        self.n_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.qkv = nn.Linear(attention_dim, attention_dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.dp_k = nn.Linear(self.head_dim, dp_rank)
        self.dp_v = nn.Linear(self.head_dim, dp_rank)
        self.ff = nn.Sequential(
            nn.Linear(attention_dim, 4 * attention_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * attention_dim, attention_dim)
        )
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        # EMA matrix
        ema_matrix = torch.zeros((512, 512))  # 512为最大patch数，可根据实际调整
        ema_matrix[0][0] = 1
        for i in range(1, 512):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i-1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix', ema_matrix)

    def ema(self, src):
        # src: (B, L, n_heads, head_dim)
        L = src.shape[1]
        if L > self.ema_matrix.shape[0]:
            # 动态扩展ema_matrix
            device = self.ema_matrix.device
            old_size = self.ema_matrix.shape[0]
            new_ema = torch.zeros((L, L), device=device)
            new_ema[:old_size, :old_size] = self.ema_matrix.data
            alpha = self.ema_matrix[1,1].item()  # 取原有alpha
            for i in range(old_size, L):
                for j in range(i):
                    new_ema[i, j] = new_ema[i-1, j] * (1 - alpha)
                new_ema[i, i] = alpha
            self.ema_matrix = nn.Parameter(new_ema, requires_grad=False)
        return torch.einsum('blhd,ga->bghd', src, self.ema_matrix[:L, :L])

    def dynamic_projection(self, src, mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum('blhd,blhc->blcd', src, src_dp)
        return src_dp

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        h = self.n_heads
        d = self.head_dim
        qkv = self.qkv(x).reshape(B, L, 3, h, d).permute(2, 0, 3, 1, 4)  # (3, B, h, L, d)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, L, d)

        # ========== 1. EMA + 动态投影注意力 ==========
        # EMA
        q_ema = self.ema(q.transpose(1,2))  # (B, L, h, d)
        k_ema = self.ema(k.transpose(1,2))
        # 动态投影
        k_dp = self.dynamic_projection(k.transpose(1,2), self.dp_k)
        v_dp = self.dynamic_projection(v.transpose(1,2), self.dp_v)
        # 注意力分数
        attn_score = torch.einsum('blhd,blcd->blhc', q_ema, k_dp) / (d ** 0.5)
        attn = self.attn_dropout(F.softmax(attn_score, dim=-2))
        out_ema_dp = torch.einsum('blhc,blcd->blhd', attn, v_dp).reshape(B, L, D)

        # ========== 2. 标准多头注意力 ==========
        # q, k, v: (B, h, L, d)
        attn_score_std = torch.einsum('bhld,bhmd->bhlm', q, k) / (d ** 0.5)  # (B, h, L, L)
        attn_std = self.attn_dropout(F.softmax(attn_score_std, dim=-1))  # (B, h, L, L)
        out_std = torch.einsum('bhlm,bhmd->bhld', attn_std, v)  # (B, h, L, d)
        out_std = out_std.permute(0, 2, 1, 3).reshape(B, L, D)  # (B, L, D)

        # ========== 3. 融合两种注意力输出 ==========
        out = out_ema_dp + out_std

        # ========== 4. 残差+归一化+FFN ==========
        x = x + out
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x
