import numpy as np
import torch

from utils.DataLoader import get_link_prediction_data
from utils.utils import get_neighbor_sampler
from models.DyGFormerD import DyGFormerD

dataset_name = 'USLegis'
node_raw_features, edge_raw_features, full_data, *_ = get_link_prediction_data(
    dataset_name=dataset_name, val_ratio=0.15, test_ratio=0.15)

full_neighbor_sampler = get_neighbor_sampler(
    data=full_data, sample_neighbor_strategy='recent', time_scaling_factor=1e-6, seed=1)

# Model WITHOUT precompute (pure online) for reference
model_online = DyGFormerD(
    node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
    neighbor_sampler=full_neighbor_sampler,
    time_feat_dim=100, channel_embedding_dim=32, patch_size=4, num_layers=2, num_heads=4,
    dropout=0.2, max_input_sequence_length=512, device='cpu',
    use_rpe=True, rwpe_dim=8, max_subgraph_size=512, rwpe_precomputed_dir=None,
)
model_online.eval()

# Model WITH the new modelfn precompute dir
precomp_dir = f'processed_data/{dataset_name}/rwpe_precomputed_modelfn_seq512_dim8_ms512'
model_precomp = DyGFormerD(
    node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
    neighbor_sampler=full_neighbor_sampler,
    time_feat_dim=100, channel_embedding_dim=32, patch_size=4, num_layers=2, num_heads=4,
    dropout=0.2, max_input_sequence_length=512, device='cpu',
    use_rpe=True, rwpe_dim=8, max_subgraph_size=512, rwpe_precomputed_dir=precomp_dir,
)
model_precomp.eval()

# Pick a batch of edges from the LATER part of the timeline (richer neighbor history,
# more likely to produce non-trivial / non-zero RWPE values) plus some random ones.
rng = np.random.RandomState(0)
n = full_data.num_interactions
late_idxs = np.arange(n - 64, n)
rand_idxs = rng.choice(n, size=64, replace=False)
idxs = np.unique(np.concatenate([late_idxs, rand_idxs]))

src = full_data.src_node_ids[idxs]
dst = full_data.dst_node_ids[idxs]
t = full_data.node_interact_times[idxs]
eid = full_data.edge_ids[idxs]

with torch.no_grad():
    # Online path (from a model that never saw any precompute dir)
    src_seq_online, dst_seq_online = model_online.compute_src_dst_node_temporal_embeddings.__wrapped__ if False else (None, None)

    # Use the internal RWPE computation directly via compute_src_dst_node_temporal_embeddings's
    # RWPE branch by monkey-probing: simplest is to call the private helpers the same way forward() does.
    src_nbr_ids_list, src_edge_ids_list, src_nbr_times_list = \
        model_online.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src, node_interact_times=t)
    dst_nbr_ids_list, dst_edge_ids_list, dst_nbr_times_list = \
        model_online.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst, node_interact_times=t)

    src_padded_ids, _, _ = model_online.pad_sequences(
        node_ids=src, node_interact_times=t, nodes_neighbor_ids_list=src_nbr_ids_list,
        nodes_edge_ids_list=src_edge_ids_list, nodes_neighbor_times_list=src_nbr_times_list,
        patch_size=model_online.patch_size, max_input_sequence_length=model_online.max_input_sequence_length)
    dst_padded_ids, _, _ = model_online.pad_sequences(
        node_ids=dst, node_interact_times=t, nodes_neighbor_ids_list=dst_nbr_ids_list,
        nodes_edge_ids_list=dst_edge_ids_list, nodes_neighbor_times_list=dst_nbr_times_list,
        patch_size=model_online.patch_size, max_input_sequence_length=model_online.max_input_sequence_length)

    online_src_pe = model_online.compute_rwpe_features(
        padded_nodes_neighbor_ids=src_padded_ids, nodes_neighbor_ids_list=src_nbr_ids_list,
        node_ids=src, node_interact_times=t)
    online_dst_pe = model_online.compute_rwpe_features(
        padded_nodes_neighbor_ids=dst_padded_ids, nodes_neighbor_ids_list=dst_nbr_ids_list,
        node_ids=dst, node_interact_times=t)

    # Precompute-lookup path: exercise the model's actual forward()-used method
    precomp_src_pe, precomp_dst_pe = model_precomp._load_rwpe_from_precomp(
        edge_ids=eid,
        src_node_ids=src, dst_node_ids=dst, node_interact_times=t,
        src_seq_len=src_padded_ids.shape[1], dst_seq_len=dst_padded_ids.shape[1],
        src_padded_nodes_neighbor_ids=src_padded_ids, dst_padded_nodes_neighbor_ids=dst_padded_ids,
        src_nodes_neighbor_ids_list=src_nbr_ids_list, dst_nodes_neighbor_ids_list=dst_nbr_ids_list,
    )

max_diff_src = (online_src_pe - precomp_src_pe).abs().max().item()
max_diff_dst = (online_dst_pe - precomp_dst_pe).abs().max().item()
exact_match_src = torch.equal(online_src_pe, precomp_src_pe)
exact_match_dst = torch.equal(online_dst_pe, precomp_dst_pe)

print(f'Sampled {len(idxs)} positive edges from {dataset_name}')
print(f'max |online_src - precomp_src| = {max_diff_src}')
print(f'max |online_dst - precomp_dst| = {max_diff_dst}')
print(f'exact match src: {exact_match_src}, dst: {exact_match_dst}')
print(f'online_src_pe.sum()={online_src_pe.sum().item():.6f} precomp_src_pe.sum()={precomp_src_pe.sum().item():.6f}')
