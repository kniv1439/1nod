import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from utils.multihead_flashdiff_2 import MultiheadFlashDiff2
from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from utils.graph import NeighborFinder, GraphMaker
from utils.wsdmmodule import FixedSinCosEncode
from utils.rotary import apply_rotary_emb

# 尝试导入flash_attn
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("WARNING: flash_attn not available, falling back to standard attention")

from utils.rms_norm import RMSNorm


def recent_interact_src_to_neigh(memory, src_idx_l, neigh_idx_l, cur_time_l, alpha=10, beta=1, eta=10000):
    most_recent_interact = np.zeros_like(neigh_idx_l)
    interact_cnt = np.zeros_like(neigh_idx_l)
    for i in range(neigh_idx_l.shape[0]):
        for j in range(neigh_idx_l.shape[1]):
            if neigh_idx_l[i, j] == 0:
                most_recent_interact[i, j] = 0.0
                interact_cnt[i, j] = 0
                continue
            ky = f'{src_idx_l[i]}_{neigh_idx_l[i, j]}'
            if ky in memory:
                most_recent_interact[i, j] = memory[ky]['recent']
                interact_cnt[i, j] = memory[ky]['count']
            else:
                most_recent_interact[i, j] = 0.0
                interact_cnt[i, j] = 0
    intensity = most_recent_interact / (interact_cnt * cur_time_l[:, np.newaxis] + 1)
    recent = (cur_time_l[:, np.newaxis] - most_recent_interact) / (cur_time_l[:, np.newaxis] + 1)
    final_dis = beta * intensity + alpha * recent
    final_dis = final_dis * eta
    return final_dis

def get_spatial_distance(src_neigh, tgt_neigh, offset):
    mask = (src_neigh[:, :, None] == tgt_neigh[:, None, :])
    min_indices = np.zeros_like(src_neigh) + 10000
    for i in range(mask.shape[-1]):
        index = np.where(mask[:, :, i] == True)
        min_indices[index[0], index[1]] = np.minimum(min_indices[index[0], index[1]], i)
    transformed = np.zeros_like(src_neigh)
    for i in range(len(offset) - 1):
        mask2 = np.logical_and(min_indices >= offset[i], min_indices < offset[i + 1])
        transformed[mask2] = i
    transformed[min_indices >= offset[-1]] = len(offset)
    transformed[src_neigh == 0] = len(offset)
    transformed = 10000 * transformed
    return transformed


class STDIFFDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_finder: NeighborFinder,
                 time_feat_dim: int, channel_embedding_dim: int, num_heads: int, num_layers: int = 28,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu', 
                 k_hop: int = 2, num_neighbors: list = [10, 5]):
        """
        STDIFFDyG model: combines STDyGFormer's k-hop sampling with DIFFDyG's processing.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_finder: neighbor finder for k-hop sampling
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param num_heads: int, number of attention heads
        :param num_layers: int, number of transformer layers
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        :param k_hop: int, k-hop sampling depth
        :param num_neighbors: list, number of neighbors for each hop
        """
        super(STDIFFDyG, self).__init__()
        
        self.num_channels = 5  # node, edge, time, neighbor_co_occurrence, spatial
        
        # ---- 重写调整逻辑：保证 num_channels * channel_embedding_dim 能被 2*num_heads 整除 ----
        import math
        required_divisor = 2 * num_heads  # 旋转位置编码要求
        lcm_unit = required_divisor // math.gcd(self.num_channels, required_divisor)
        if channel_embedding_dim % lcm_unit != 0:
            channel_embedding_dim = ((channel_embedding_dim + lcm_unit - 1) // lcm_unit) * lcm_unit
        
        # 如果没有flash_attn，自动调整参数以节省资源
        if not FLASH_ATTN_AVAILABLE:
            # 减少Transformer层数
            num_layers = min(num_layers, 6)
            # 减少最大序列长度
            max_input_sequence_length = min(max_input_sequence_length, 128)
            
        self.channel_embedding_dim = channel_embedding_dim
        self.num_heads = num_heads

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_finder = neighbor_finder
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        self.k_hop = k_hop

        # -------------------------------------------------------------------
        # 兼容传入的 num_neighbors 为 int / list / tuple / str 的各种情况
        # 目标：确保 self.num_neighbors 一定是长度 == k_hop 的列表，
        #       这样在 GraphMaker(num_neigh=self.num_neighbors, ...) 里才能安全下标。
        # -------------------------------------------------------------------
        if isinstance(num_neighbors, str):
            import ast
            num_neighbors = ast.literal_eval(num_neighbors)

        if isinstance(num_neighbors, int):
            num_neighbors = [num_neighbors] * k_hop
        elif isinstance(num_neighbors, (list, tuple)):
            num_neighbors = list(num_neighbors)
            if len(num_neighbors) < k_hop:
                # 不足时用最后一个值填充
                num_neighbors.extend([num_neighbors[-1]] * (k_hop - len(num_neighbors)))
            else:
                num_neighbors = num_neighbors[:k_hop]
        else:
            raise ValueError(f"num_neighbors 必须为 int / list / tuple / str，但收到 {type(num_neighbors)}")

        self.num_neighbors = num_neighbors

        # STDyGFormer风格的编码器
        self.temporal_encoder = FixedSinCosEncode(hidden_size=self.time_feat_dim)
        self.spatial_encoder = FixedSinCosEncode(hidden_size=self.time_feat_dim)

        # 邻居共现编码器
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, 
            device=self.device
        )

        # 5个通道的投影层
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'spatial': nn.Linear(in_features=self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        # GraphMaker for k-hop sampling
        self.graph_maker = GraphMaker(num_neigh=self.num_neighbors, max_depth=self.k_hop)
        self.memory = {}

        self.attention_dim = self.num_channels * self.channel_embedding_dim
        # 确保数学关系正确
        if (self.attention_dim // self.num_heads) % 2 != 0:
            raise ValueError(f"attention_dim ({self.attention_dim}) must be divisible by 2*num_heads ({2*self.num_heads}). "
                            f"Please adjust channel_embedding_dim or num_heads.")
        self.head_dim = self.attention_dim // self.num_heads // 2
        
        # DIFFDyG风格的transformer
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
        """
        Compute rotary embeddings (cosine and sine) for RoPE.
        """
        # 使用更大的值来确保足够长
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
                                               node_interact_times: np.ndarray, cos=None, sin=None, rotary_dim=None):
        """
        Compute source and destination node temporal embeddings using k-hop sampling.
        """
        # ---------------------------------------------------------------
        # 安全转换输入类型：若上游传入的是 GPU Tensor，则转为 CPU numpy
        # ---------------------------------------------------------------
        if isinstance(src_node_ids, torch.Tensor):
            src_node_ids = src_node_ids.detach().cpu().numpy().astype(np.int32)
        if isinstance(dst_node_ids, torch.Tensor):
            dst_node_ids = dst_node_ids.detach().cpu().numpy().astype(np.int32)
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy().astype(np.float32)

        # k-hop采样 - STDyGFormer风格
        src_one_hop_node, src_one_hop_t, src_node_records, src_t_records, src_eidx_g, src_offset_per_layer = \
            self.neighbor_finder.find_k_hop(self.k_hop, src_node_ids, node_interact_times, self.graph_maker, self.num_neighbors)
        
        dst_one_hop_node, dst_one_hop_t, dst_node_records, dst_t_records, dst_eidx_g, dst_offset_per_layer = \
            self.neighbor_finder.find_k_hop(self.k_hop, dst_node_ids, node_interact_times, self.graph_maker, self.num_neighbors)

        # 获取节点特征
        src_node_features = self.node_raw_features[torch.from_numpy(src_node_records).long()]
        dst_node_features = self.node_raw_features[torch.from_numpy(dst_node_records).long()]
        
        # 获取边特征（取主对角线）
        # numpy.diagonal 默认产出 float64，需要转成 int64 作为索引
        src_diag_idx = np.diagonal(src_eidx_g, axis1=1, axis2=2).astype(np.int64)
        dst_diag_idx = np.diagonal(dst_eidx_g, axis1=1, axis2=2).astype(np.int64)

        idx_device = self.edge_raw_features.device
        src_edge_idx_tensor = torch.from_numpy(src_diag_idx).long().to(idx_device)
        dst_edge_idx_tensor = torch.from_numpy(dst_diag_idx).long().to(idx_device)

        src_edge_features = self.edge_raw_features[src_edge_idx_tensor]
        dst_edge_features = self.edge_raw_features[dst_edge_idx_tensor]
        
        # 计算时间特征
        src_temporal_feature = self.temporal_encoder(
            recent_interact_src_to_neigh(self.memory, src_node_ids, src_node_records, node_interact_times)
        )
        dst_temporal_feature = self.temporal_encoder(
            recent_interact_src_to_neigh(self.memory, dst_node_ids, dst_node_records, node_interact_times)
        )
        
        # 计算空间特征
        src_spatial_feature = self.spatial_encoder(
            get_spatial_distance(src_node_records, src_node_records, src_offset_per_layer)
        )
        dst_spatial_feature = self.spatial_encoder(
            get_spatial_distance(dst_node_records, dst_node_records, dst_offset_per_layer)
        )
        
        # 计算邻居共现特征
        src_neighbor_co_occurrence_features, dst_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(
                src_padded_nodes_neighbor_ids=src_node_records,
                dst_padded_nodes_neighbor_ids=dst_node_records
            )

        # 将全部特征转到同一设备，避免 CPU/GPU 混用
        device = self.device
        src_node_features = src_node_features.to(device)
        dst_node_features = dst_node_features.to(device)
        src_edge_features = src_edge_features.to(device)
        dst_edge_features = dst_edge_features.to(device)
        src_temporal_feature = src_temporal_feature.to(device)
        dst_temporal_feature = dst_temporal_feature.to(device)
        src_spatial_feature  = src_spatial_feature.to(device)
        dst_spatial_feature  = dst_spatial_feature.to(device)
        src_neighbor_co_occurrence_features = src_neighbor_co_occurrence_features.to(device)
        dst_neighbor_co_occurrence_features = dst_neighbor_co_occurrence_features.to(device)

        # 统一转换 dtype 为 float32，防止 Double/Float 混用
        src_node_features  = src_node_features.float()
        dst_node_features  = dst_node_features.float()
        src_edge_features  = src_edge_features.float()
        dst_edge_features  = dst_edge_features.float()
        src_temporal_feature = src_temporal_feature.float()
        dst_temporal_feature = dst_temporal_feature.float()
        src_spatial_feature  = src_spatial_feature.float()
        dst_spatial_feature  = dst_spatial_feature.float()
        src_neighbor_co_occurrence_features = src_neighbor_co_occurrence_features.float()
        dst_neighbor_co_occurrence_features = dst_neighbor_co_occurrence_features.float()

        # 投影5个通道的特征
        src_features = torch.cat([
            self.projection_layer['node'](src_node_features),
            self.projection_layer['edge'](src_edge_features),
            self.projection_layer['time'](src_temporal_feature),
            self.projection_layer['spatial'](src_spatial_feature),
            self.projection_layer['neighbor_co_occurrence'](src_neighbor_co_occurrence_features)
        ], dim=-1)

        dst_features = torch.cat([
            self.projection_layer['node'](dst_node_features),
            self.projection_layer['edge'](dst_edge_features),
            self.projection_layer['time'](dst_temporal_feature),
            self.projection_layer['spatial'](dst_spatial_feature),
            self.projection_layer['neighbor_co_occurrence'](dst_neighbor_co_occurrence_features)
        ], dim=-1)

        # 计算rotary embeddings
        padded_seq_len = max(src_features.shape[1], dst_features.shape[1])
        embed_dim = src_features.shape[-1]
        assert embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        
        # 计算 headdim_total 和 rotary_dim
        headdim_total = embed_dim // self.num_heads
        if rotary_dim is None:
            rotary_dim = min((embed_dim // 2) * 2, headdim_total)
        
        # 确保cos和sin的长度足够，如果未提供则计算新的
        if cos is None or sin is None:
            cos, sin = self.compute_rotary_embedding(padded_seq_len, rotary_dim)

        # 通过transformer层 - DIFFDyG风格
        for transformer in self.transformers:
            src_features = transformer(src_features, cos, sin, rotary_dim)
            dst_features = transformer(dst_features, cos, sin, rotary_dim)

        # 聚合特征
        src_node_embeddings = self.output_layer(torch.mean(src_features, dim=1))
        dst_node_embeddings = self.output_layer(torch.mean(dst_features, dim=1))

        return src_node_embeddings, dst_node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler):
        """兼容外部调用：
        - 训练阶段模型以 NeighborFinder 为核心，
        - 外部管线在验证/测试时会调用 set_neighbor_sampler(full_neighbor_finder)。

        如果传进来的是 NeighborFinder，就直接替换；
        如果仍旧是 NeighborSampler（旧模型用），则忽略。
        """
        from utils.graph import NeighborFinder
        if isinstance(neighbor_sampler, NeighborFinder):
            self.neighbor_finder = neighbor_sampler

    def forward(self, batch):
        """
        Forward pass for the model
        """
        src_node_ids = batch['src_node_ids']
        dst_node_ids = batch['dst_node_ids']
        node_interact_times = batch['node_interact_times']
        
        src_emb, dst_emb = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times
        )
        return src_emb, dst_emb


class TransformerDecoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, head_dim: int, depth: int, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.multi_head_attention = MultiheadFlashDiff2(
            embed_dim=attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=3 * attention_dim),
            nn.Linear(in_features=(3 * attention_dim) // 2, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            RMSNorm(attention_dim),
            RMSNorm(attention_dim)
        ])

    def forward(self, src_features, cos, sin, rotary_dim):
        # 仅在flash_attn可用时将输入特征转换为fp16
        if FLASH_ATTN_AVAILABLE:
            src_features = src_features.half()

        # Transpose inputs for multi-head attention
        transposed_inputs = src_features.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        transposed_inputs = self.norm_layers[0](transposed_inputs)

        # 确保输入为四维 [seq_len, batch_size, num_heads, head_dim]
        seq_len, batch_size, embed_dim = transposed_inputs.shape
        head_dim = self.head_dim
        assert embed_dim == self.num_heads * 2 * head_dim, \
            f"Embedding dimension ({embed_dim}) must be {self.num_heads * 2 * head_dim}"
        
        transposed_inputs = transposed_inputs.view(seq_len, batch_size, self.num_heads, 2 * head_dim)

        # 动态调整 cos 和 sin 的长度
        if cos.shape[0] < seq_len:
            raise ValueError(f"cos and sin length ({cos.shape[0]}) must be >= input sequence length ({seq_len})")
        cos = cos[:seq_len, :rotary_dim // 2]
        sin = sin[:seq_len, :rotary_dim // 2]

        # 应用旋转嵌入
        transposed_inputs = apply_rotary_emb(transposed_inputs, cos, sin)

        # 恢复为三维 [seq_len, batch_size, embed_dim] 以适配多头注意力层
        transposed_inputs = transposed_inputs.reshape(seq_len, batch_size, embed_dim)

        # Multi-head attention
        hidden_states = self.multi_head_attention(
            x=transposed_inputs,
            rel_pos=(cos, sin),
            attn_mask=None
        ).transpose(0, 1)

        # Residual connection and feedforward
        outputs = src_features + self.dropout(hidden_states)
        hidden_states = self.linear_layers[0](self.norm_layers[1](outputs))
        x1, x2 = torch.chunk(hidden_states, chunks=2, dim=-1)
        swiglu_output = F.gelu(x1) * torch.sigmoid(x2)
        hidden_states = self.linear_layers[1](self.dropout(swiglu_output))
        outputs = outputs + self.dropout(hidden_states)
        return outputs


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
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        """
        # two lists to store the appearances of source and destination nodes
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

        # set the appearances of the padded node (with zero index) to zeros
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

        # sum the neighbor co-occurrence features
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features 