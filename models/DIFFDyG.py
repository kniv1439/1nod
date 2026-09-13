import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from utils.multihead_flashdiff_2 import MultiheadFlashDiff2NoRoPE
from models.modules import TimeEncoder
from utils.utils import NeighborSampler

# 尝试导入flash_attn
try:
    from flash_attn import flash_attn_func # type: ignore
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("WARNING: flash_attn not available, falling back to standard attention")

from utils.rms_norm import RMSNorm


class DIFFDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, num_heads: int, num_layers: int = 28,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu', patch_size: int = 4):
        """
        DIFFDyG model with patching.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param num_heads: int, number of attention heads
        :param num_layers: int, number of transformer layers
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        :param patch_size: int, patch size (default: 4)
        """
        super(DIFFDyG, self).__init__()
        
        self.num_channels = 4  # node, edge, time, neighbor_co_occurrence
        
        # 确保 channel_embedding_dim 满足条件：num_channels * channel_embedding_dim 可以被 2 * num_heads 整除
        if (self.num_channels * channel_embedding_dim) % (2 * num_heads) != 0:
            # 计算调整后的 channel_embedding_dim
            required_divisor = 2 * num_heads
            remainder = (self.num_channels * channel_embedding_dim) % required_divisor
            if remainder > 0:
                # 向上取整到最接近的满足条件的值
                channel_embedding_dim += (required_divisor - remainder) // self.num_channels
                if (self.num_channels * channel_embedding_dim) % required_divisor != 0:
                    # 如果仍然不能整除，再增加一些
                    channel_embedding_dim += required_divisor // self.num_channels
            
        # 如果没有flash_attn，自动调整参数以节省资源
        if not FLASH_ATTN_AVAILABLE:
            # 减少Transformer层数
            num_layers = min(num_layers, 6)  # 更激进地减少层数
            # 减少最大序列长度
            max_input_sequence_length = min(max_input_sequence_length, 128)  # 更小的序列长度
            
        self.channel_embedding_dim = channel_embedding_dim
        self.num_heads = num_heads
        self.patch_size = patch_size

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

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # 添加邻居共现编码器
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        # 修改 projection_layer 的输入维度以支持 patching
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.attention_dim = self.num_channels * self.channel_embedding_dim
        # 移除断言，确保数学关系正确
        if (self.attention_dim // self.num_heads) % 2 != 0:
            raise ValueError(f"attention_dim ({self.attention_dim}) must be divisible by 2*num_heads ({2*self.num_heads}). "
                            f"Please adjust channel_embedding_dim or num_heads.")
        self.head_dim = self.attention_dim // self.num_heads // 2
        self.transformers = nn.ModuleList([
            TransformerDecoder(
                attention_dim=self.attention_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,  # 显式传递 head_dim
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

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, **kwargs):
        """
        Compute source and destination node temporal embeddings with patching (without RoPE).
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param **kwargs: 额外的关键字参数（如 record_last_attn），用于兼容训练脚本，但不使用
        :return:
        """
        # Get the first-hop neighbors of source and destination nodes
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # Pad the sequences of first-hop neighbors for source and destination nodes
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # Get the features of the sequence of source and destination nodes
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # 计算邻居共现特征
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # 获取 patches
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

        # Align the patch encoding dimension
        src_features = torch.cat([
            self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features),
            self.projection_layer['edge'](src_patches_nodes_edge_raw_features),
            self.projection_layer['time'](src_patches_nodes_neighbor_time_features),
            self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)
        ], dim=-1)

        dst_features = torch.cat([
            self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features),
            self.projection_layer['edge'](dst_patches_nodes_edge_raw_features),
            self.projection_layer['time'](dst_patches_nodes_neighbor_time_features),
            self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)
        ], dim=-1)

        # Pass through transformers (without RoPE)
        for transformer in self.transformers:
            src_features = transformer(src_features)
            dst_features = transformer(dst_features)

        # Aggregate features (对 patches 进行平均)
        src_node_embeddings = self.output_layer(torch.mean(src_features, dim=1))
        dst_node_embeddings = self.output_layer(torch.mean(dst_features, dim=1))

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_input_sequence_length: int = 256):
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            # 截断到 max_input_sequence_length - 1
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            current_len = len(nodes_neighbor_ids_list[idx])
            if current_len > max_seq_length:
                max_seq_length = current_len
        
        # 包含目标节点本身
        max_seq_length += 1
        
        # 确保 max_seq_length 是 patch_size 的倍数
        if max_seq_length % self.patch_size != 0:
            max_seq_length += (self.patch_size - max_seq_length % self.patch_size)
        assert max_seq_length % self.patch_size == 0
        
        # 使用成员变量的最大序列长度
        max_seq_length = min(max_seq_length, self.max_input_sequence_length)
        
        # 为节省内存，设置一个安全上限
        if not FLASH_ATTN_AVAILABLE and max_seq_length > 64:
            # 确保是 patch_size 的倍数
            max_seq_length = ((64 + self.patch_size - 1) // self.patch_size) * self.patch_size
            
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length), dtype=np.float32)
        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
            if len(nodes_neighbor_ids_list[idx]) > 0:
                # 确保不会超出边界
                neighbor_len = min(len(nodes_neighbor_ids_list[idx]), max_seq_length - 1)
                # 如果序列太长，再次截断（安全检查）
                if neighbor_len < len(nodes_neighbor_ids_list[idx]):
                    nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-neighbor_len:]
                    nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-neighbor_len:]
                    nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-neighbor_len:]
                
                padded_nodes_neighbor_ids[idx, 1:1 + neighbor_len] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1:1 + neighbor_len] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1:1 + neighbor_len] = nodes_neighbor_times_list[idx]
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        
        """
        Get node, edge, and time features.
        """
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
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        # 关键：全部转到 self.device
        padded_nodes_neighbor_node_raw_features = padded_nodes_neighbor_node_raw_features.to(self.device)
        padded_nodes_edge_raw_features = padded_nodes_edge_raw_features.to(self.device)
        padded_nodes_neighbor_time_features = padded_nodes_neighbor_time_features.to(self.device)

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor, patch_size: int = 1):
        """
        Get the sequence of patches for nodes.
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, feat_dim)
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
        # Tensor, shape (batch_size, num_patches, patch_size * neighbor_co_occurrence_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        Set neighbor sampler and reset random state.
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def forward(self, batch):
        # 这里 batch 通常是一个 dict，包含 src_node_ids, dst_node_ids, node_interact_times 等
        # 你可以仿照 NPTDyG 或其他模型的 forward 实现
        src_node_ids = batch['src_node_ids']
        dst_node_ids = batch['dst_node_ids']
        node_interact_times = batch['node_interact_times']
        # 假设你有 compute_src_dst_node_temporal_embeddings 方法
        src_emb, dst_emb = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times
        )
        return src_emb, dst_emb


class TransformerDecoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, head_dim: int, depth: int, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim  # 保存 head_dim 为类属性
        self.multi_head_attention = MultiheadFlashDiff2NoRoPE(
            embed_dim=attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,  # 显式传递 head_dim
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

    def forward(self, src_features):
        # 保存原始 dtype
        original_dtype = src_features.dtype
        
        # 不在这里转换输入为 FP16，让投影层在 FP32 下计算以提高数值稳定性
        # 只在调用 flash_attn_func 时临时转换
        # if FLASH_ATTN_AVAILABLE:
        #     src_features = src_features.half()

        # Transpose inputs for multi-head attention
        transposed_inputs = src_features.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        transposed_inputs = self.norm_layers[0](transposed_inputs)

        # 确保输入为四维 [seq_len, batch_size, num_heads, head_dim]
        seq_len, batch_size, embed_dim = transposed_inputs.shape
        head_dim = self.head_dim  # 从初始化参数获取
        assert embed_dim == self.num_heads * 2 * head_dim, \
            f"Embedding dimension ({embed_dim}) must be {self.num_heads * 2 * head_dim}"
        transposed_inputs = transposed_inputs.view(seq_len, batch_size, self.num_heads, 2 * head_dim)

        # 恢复为三维 [seq_len, batch_size, embed_dim] 以适配多头注意力层
        transposed_inputs = transposed_inputs.reshape(seq_len, batch_size, embed_dim)

        # Multi-head attention (without RoPE)
        hidden_states = self.multi_head_attention(
            x=transposed_inputs,  # 输入是三维 [seq_len, batch_size, embed_dim]
            rel_pos=None,  # 不使用 RoPE
            attn_mask=None
        ).transpose(0, 1)
        
        # 确保 hidden_states 与 src_features 类型一致
        if hidden_states.dtype != src_features.dtype:
            hidden_states = hidden_states.to(dtype=src_features.dtype)

        # Residual connection and feedforward
        outputs = src_features + self.dropout(hidden_states)
        hidden_states = self.linear_layers[0](self.norm_layers[1](outputs))
        x1, x2 = torch.chunk(hidden_states, chunks=2, dim=-1)
        swiglu_output = F.gelu(x1) * torch.sigmoid(x2)
        hidden_states = self.linear_layers[1](self.dropout(swiglu_output))
        outputs = outputs + self.dropout(hidden_states)
        
        # 确保返回类型与原始输入一致
        if outputs.dtype != original_dtype:
            outputs = outputs.to(dtype=original_dtype)
        
        return outputs


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