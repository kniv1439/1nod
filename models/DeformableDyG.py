import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils.RevIN2 import RevIN
from utils.Global_Attn import DAttentionBaseline
from utils.Transformer_Module import TransformerMLPWithConv, TransformerMLP, LayerNormProxy
from utils.head import Flatten_Head
from timm.models.layers import DropPath
from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        return x * self.weight.view(-1, 1)


class Stage(nn.Module):

    def __init__(self,
                 fmap_size,
                 dim_embed, depths,
                 drop_path_rate, layer_scale_value,
                 use_pe,
                 use_lpu,local_kernel_size,
                 expansion, drop, use_dwc_mlp,
                 heads, attn_drop, proj_drop,
                 stage_spec,
                 window_size,
                 nat_ksize,
                 ksize, stride,
                 n_groups, offset_range_factor, no_off,
                 dwc_pe, fixed_pe, log_cpb,
                 ):


        super(Stage,self).__init__()
        fmap_size = fmap_size
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu


        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for d in range(2 * depths)]
        )


        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv1d(dim_embed, dim_embed, kernel_size=local_kernel_size, stride=1, padding=local_kernel_size//2,
                          groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )


        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()

        for i in range(depths):

            if stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, ksize, log_cpb)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')


            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())


        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop, local_kernel_size) for _ in range(depths)
            ]
        )

        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )

    def forward(self, x):


        for d in range(self.depths):

            if self.use_lpu:
                x0 = x.clone()
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'D':

                x0 = x.clone()
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0

                x0 = x.clone()
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x



class DeformableDyG(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 4, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2, max_input_sequence_length: int = 512, device: str = 'cpu'):
        """
        DeformableDyG model for dynamic graph link prediction and node classification.
        :param patch_size: int, patch size for dividing neighbor sequences
        """
        super(DeformableDyG, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        # 时间编码器
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        
        # 添加邻居共现编码器
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        # 特征投影层 - 输入维度变为 patch_size * feat_dim
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4  # node, edge, time, neighbor_co_occurrence

        # 主干网络
        self.stages = nn.ModuleList([
            Stage(
                fmap_size=max_input_sequence_length,
                dim_embed=channel_embedding_dim * 4,
                depths=num_layers,
                drop_path_rate=[dropout] * num_layers,
                layer_scale_value=1e-5,
                use_pe=True,
                use_lpu=True,
                local_kernel_size=3,
                expansion=4,
                drop=dropout,
                use_dwc_mlp=False,
                heads=num_heads,
                attn_drop=dropout,
                proj_drop=dropout,
                stage_spec=['D'] * num_layers,
                window_size=8,
                nat_ksize=3,
                ksize=3,
                stride=1,
                n_groups=1,
                offset_range_factor=1,
                no_off=False,
                dwc_pe=False,
                fixed_pe=False,
                log_cpb=False
            )
        ])

        # 输出层
        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

   

    def forward(self, *args, **kwargs):
        # 支持 batch 字典输入
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            src_node_ids = batch['src_node_ids']
            dst_node_ids = batch['dst_node_ids']
            node_interact_times = batch['node_interact_times']
            return self.compute_src_dst_node_temporal_embeddings(src_node_ids, dst_node_ids, node_interact_times)
        else:
            # 兼容原有方式
            return self.compute_src_dst_node_temporal_embeddings(*args, **kwargs)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors=None):
        """
        Compute source and destination node temporal embeddings.
        Note: num_neighbors parameter is included for interface compatibility but not used,
        as this model uses get_all_first_hop_neighbors internally.
        """
        # 获取源节点和目标节点的一阶邻居
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # 对邻居序列进行填充
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # 获取源节点和目标节点的特征
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

        # 【新增】将特征划分为 patches
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

        # 对 patch 特征进行投影并拼接
        # shape: (batch_size, num_patches, channel_embedding_dim)
        src_features = torch.cat([
            self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features),
            self.projection_layer['edge'](src_patches_nodes_edge_raw_features),
            self.projection_layer['time'](src_patches_nodes_neighbor_time_features),
            self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)
        ], dim=-1)  # [batch, num_patches, channel_embedding_dim*4]

        dst_features = torch.cat([
            self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features),
            self.projection_layer['edge'](dst_patches_nodes_edge_raw_features),
            self.projection_layer['time'](dst_patches_nodes_neighbor_time_features),
            self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)
        ], dim=-1)  # [batch, num_patches, channel_embedding_dim*4]

        # 通过deformable stages处理
        src_features = src_features.transpose(1, 2)  # [batch, channel_embedding_dim*4, num_patches]
        dst_features = dst_features.transpose(1, 2)  # [batch, channel_embedding_dim*4, num_patches]
        
        for stage in self.stages:
            src_features = stage(src_features)
            dst_features = stage(dst_features)
        
        # 转换回原始维度并应用输出层
        src_features = src_features.transpose(1, 2)  # [batch, num_patches, channel_embedding_dim*4]
        dst_features = dst_features.transpose(1, 2)  # [batch, num_patches, channel_embedding_dim*4]
        
        # 对 patch 维度取平均，得到节点级别的嵌入
        src_node_embeddings = torch.mean(src_features, dim=1)  # [batch, channel_embedding_dim*4]
        dst_node_embeddings = torch.mean(dst_features, dim=1)  # [batch, channel_embedding_dim*4]
        
        # 应用输出层
        src_node_embeddings = self.output_layer(src_node_embeddings)  # [batch, node_feat_dim]
        dst_node_embeddings = self.output_layer(dst_node_embeddings)  # [batch, node_feat_dim]

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_input_sequence_length: int = 256):
        """
        Pad the sequences for nodes in node_ids.
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        
        max_seq_length = max(len(nodes_neighbor_ids_list[idx]) for idx in range(len(nodes_neighbor_ids_list))) + 1
        max_seq_length = max(max_seq_length, self.max_input_sequence_length)

        # 修改：确保序列长度能被 patch_size 整除
        if max_seq_length % self.patch_size != 0:
            max_seq_length += (self.patch_size - max_seq_length % self.patch_size)
        assert max_seq_length % self.patch_size == 0

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, :len(nodes_neighbor_ids_list[idx])] = nodes_neighbor_ids_list[idx]
            padded_nodes_edge_ids[idx, :len(nodes_edge_ids_list[idx])] = nodes_edge_ids_list[idx]
            padded_nodes_neighbor_times[idx, :len(nodes_neighbor_times_list[idx])] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        Get node, edge, and time features.
        """
        # 保证两者都是 tensor，并确保数据类型为 float32（与模型参数一致）
        if isinstance(node_interact_times, np.ndarray):
            # 确保转换为 float32，避免 Double (float64) 和 Float (float32) 不匹配
            node_interact_times = torch.from_numpy(node_interact_times.astype(np.float32)).to(self.device)
        elif isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.float().to(self.device)
        
        if isinstance(padded_nodes_neighbor_times, np.ndarray):
            # 确保转换为 float32
            padded_nodes_neighbor_times = torch.from_numpy(padded_nodes_neighbor_times.astype(np.float32)).to(self.device)
        elif isinstance(padded_nodes_neighbor_times, torch.Tensor):
            padded_nodes_neighbor_times = padded_nodes_neighbor_times.float().to(self.device)

        # 保证 node_interact_times 是二维
        if node_interact_times.dim() == 1:
            node_interact_times = node_interact_times.unsqueeze(1)

        timestamps = node_interact_times - padded_nodes_neighbor_times
        padded_nodes_neighbor_time_features = time_encoder(timestamps=timestamps)

        # 获取节点特征（确保索引是 long 类型）
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids.astype(np.int64))]
        # 获取边特征（确保索引是 long 类型）
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids.astype(np.int64))]
        # 将填充的部分时间特征置为 0
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, 
                    padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, 
                    padded_nodes_neighbor_co_occurrence_features: torch.Tensor,
                    patch_size: int = 1):
        """
        Get the sequence of patches for nodes.
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return: four tensors with patched features
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        patches_nodes_neighbor_node_raw_features = []
        patches_nodes_edge_raw_features = []
        patches_nodes_neighbor_time_features = []
        patches_nodes_neighbor_co_occurrence_features = []

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
            assert self.neighbor_sampler.seed is not None, "Neighbor sampler requires a seed for reproducibility."
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

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()

        n_vars = configs.n_vars
        rev = configs.revin
        revin_affine = configs.revin_affine
        revin_subtract_last = configs.revin_subtract_last

        stem_ratio = configs.stem_ratio
        down_ratio = configs.down_ratio

        fmap_size = configs.fmap_size

        dims = configs.dims
        depths = configs.depths

        drop_path_rate = configs.drop_path_rate
        layer_scale_value = configs.layer_scale_value

        use_pe = configs.use_pe

        use_lpu = configs.use_lpu
        local_kernel_size = configs.local_kernel_size

        expansion = configs.expansion
        drop = configs.drop
        use_dwc_mlp = configs.use_dwc_mlp

        heads = configs.heads
        attn_drop = configs.attn_drop
        proj_drop = configs.proj_drop

        stage_spec = configs.stage_spec

        window_size = configs.window_size
        nat_ksize = configs.nat_ksize

        ksize = configs.ksize
        stride = configs.stride
        n_groups = configs.n_groups
        offset_range_factor = configs.offset_range_factor
        no_off = configs.no_off

        dwc_pe = configs.dwc_pe
        fixed_pe = configs.fixed_pe
        log_cpb = configs.log_cpb

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        head_dropout = configs.head_dropout
        head_type = configs.head_type
        use_head_norm = configs.use_head_norm


        self.model = DeformableDyG(
            n_vars, rev, revin_affine, revin_subtract_last,
            stem_ratio,
            down_ratio,
            fmap_size,
            dims, depths,
            drop_path_rate, layer_scale_value,
            use_pe,
            use_lpu,
            local_kernel_size,
            expansion, drop, use_dwc_mlp,
            heads, attn_drop, proj_drop,
            stage_spec,
            window_size,
            nat_ksize,
            ksize, stride,
            n_groups, offset_range_factor, no_off,
            dwc_pe, fixed_pe, log_cpb,
            seq_len, pred_len, head_dropout,
            head_type, use_head_norm,
        )

    def forward(self,x):

        return self.model(x)

