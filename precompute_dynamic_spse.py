#!/usr/bin/env python3
"""
precompute_dynamic_spse.py

Precompute SPSE (Simple Path Structural Encoding) for dynamic graph datasets.

For each interaction (src, dst, t, edge_idx):
  1. Build the padded sequences for src and dst (matching pad_sequences logic)
  2. Collect ALL nodes from ALL patches (not just patch representatives)
  3. Build the temporal subgraph: edges among these nodes with timestamp < t
  4. Run get_simple_paths_count on the subgraph
  5. Aggregate path counts per patch pair using sum
  6. Store as sparse COO (pre-aggregated, patch-level)

Default SPSE parameters:
  dist_len=16, n_root_nodes_fact=0.4, min_root_nodes=4,
  max_bfs_depth=2, max_tries_per_depth=7, aggregation=sum

Storage format (per edge_idx, as value in chunk dict):
  None  →  subgraph has no edges (c_spse stays zero)
  dict  →  {
      'src_P': np.int16,          # number of src patches
      'dst_P': np.int16,          # number of dst patches
      'row':   np.int16 array,    # (nnz,)  patch indices in [0, src_P+dst_P)
      'col':   np.int16 array,    # (nnz,)
      'vals':  np.float16 array,  # (nnz, dist_len)  ← already log1p-transformed
  }

  NOTE: vals are stored as log1p(path_count) to avoid float16 overflow.
  At training time, do NOT apply log1p again — fill c_spse_cpu directly with vals.

Training-time loading:
  Load all chunk_*.pkl files into one dict {edge_idx: entry}.
  In compute_spse_bias, replace the online SPSE call with a lookup:

      entry = precomputed[int(edge_id)]
      if entry is None: continue
      src_P_pre = int(entry['src_P'])
      rows  = entry['row']
      cols  = entry['col']
      vals  = entry['vals'].astype(np.float32)   # (nnz, dist_len), already log1p
      # map dst-patch local indices to batch-combined absolute indices
      batch_rows = np.where(rows < src_P_pre, rows, src_P_batch + (rows - src_P_pre))
      batch_cols = np.where(cols < src_P_pre, cols, src_P_batch + (cols - src_P_pre))
      L_batch = src_P_batch + dst_P_batch
      ok = (batch_rows < L_batch) & (batch_cols < L_batch)
      c_spse_cpu[b, batch_rows[ok], batch_cols[ok], :] = vals[ok]
      # IMPORTANT: skip the torch.log1p(c_spse) step that follows in the online path

Usage:
  python precompute_dynamic_spse.py --dataset_name wikipedia
  python precompute_dynamic_spse.py --dataset_name reddit --chunk_size 5000
"""

import os
import sys
import json
import time
import argparse
import pickle
import math
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Worker-process globals (populated via fork / initializer)
# ---------------------------------------------------------------------------

_G_node_nbr_ids   = None
_G_node_nbr_times = None
_G_args           = None


def _init_worker(nbr_ids, nbr_times, args):
    global _G_node_nbr_ids, _G_node_nbr_times, _G_args
    _G_node_nbr_ids   = nbr_ids
    _G_node_nbr_times = nbr_times
    _G_args           = args

# ---------------------------------------------------------------------------
# Import get_simple_paths_count from Graph-SPSE-Encoding
# ---------------------------------------------------------------------------
_SPSE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Graph-SPSE-Encoding')
if _SPSE_DIR not in sys.path:
    sys.path.insert(0, _SPSE_DIR)

try:
    from count_self_avoiding_paths import get_simple_paths_count
except ImportError as e:
    raise ImportError(
        f"Cannot import get_simple_paths_count from {_SPSE_DIR}. "
        "Make sure Graph-SPSE-Encoding/ exists in the project root."
    ) from e


# ---------------------------------------------------------------------------
# Neighbor structure
# ---------------------------------------------------------------------------

def build_neighbor_structure(src_ids: np.ndarray, dst_ids: np.ndarray,
                              timestamps: np.ndarray):
    """
    Build bidirectional temporal neighbor structure matching NeighborSampler.

    For each edge (u, v, t): add v to u's history at time t, and u to v's history at time t.
    Each node's history is sorted by timestamp ascending (matching NeighborSampler.__init__).

    Returns:
        node_nbr_ids   : list[np.ndarray[int64]]   indexed by node id
        node_nbr_times : list[np.ndarray[float64]] indexed by node id
    """
    max_node = int(max(src_ids.max(), dst_ids.max()))
    raw_ids   = [[] for _ in range(max_node + 1)]
    raw_times = [[] for _ in range(max_node + 1)]

    for u, v, t in zip(src_ids.tolist(), dst_ids.tolist(), timestamps.tolist()):
        raw_ids[u].append(v);  raw_times[u].append(t)
        raw_ids[v].append(u);  raw_times[v].append(t)

    node_nbr_ids   = []
    node_nbr_times = []
    for u in range(max_node + 1):
        if raw_ids[u]:
            arr_ids   = np.array(raw_ids[u],   dtype=np.int64)
            arr_times = np.array(raw_times[u], dtype=np.float64)
            order = np.argsort(arr_times, kind='stable')
            node_nbr_ids.append(arr_ids[order])
            node_nbr_times.append(arr_times[order])
        else:
            node_nbr_ids.append(np.empty(0, dtype=np.int64))
            node_nbr_times.append(np.empty(0, dtype=np.float64))

    return node_nbr_ids, node_nbr_times


# ---------------------------------------------------------------------------
# Padded sequence + patch node extraction
# ---------------------------------------------------------------------------

def get_patch_nodes(node_id: int, t: float,
                    node_nbr_ids, node_nbr_times,
                    max_seq_len: int, patch_size: int):
    """
    Replicate pad_sequences logic for a single node at time t.

    Sequence layout:
        position 0          : node_id itself
        positions 1 .. k    : historical neighbors before t, sorted ascending by time,
                              keeping the most recent (max_seq_len - 1)
        positions k+1 ..    : zero padding up to multiple of patch_size

    Returns:
        patch_nodes : list[np.ndarray[int64]], length = num_patches
                      each array contains the non-zero node IDs in that patch
        num_patches : int
    """
    if node_id < len(node_nbr_ids):
        nbr_ids   = node_nbr_ids[node_id]
        nbr_times = node_nbr_times[node_id]
        # searchsorted with side='left' matches find_neighbors_before (strict <)
        cutoff = int(np.searchsorted(nbr_times, t, side='left'))
        nbrs = nbr_ids[:cutoff][-(max_seq_len - 1):]
    else:
        nbrs = np.empty(0, dtype=np.int64)

    actual_len  = 1 + len(nbrs)
    remainder   = actual_len % patch_size
    padded_len  = actual_len if remainder == 0 else actual_len + (patch_size - remainder)
    num_patches = padded_len // patch_size

    seq = np.zeros(padded_len, dtype=np.int64)
    seq[0] = node_id
    if len(nbrs) > 0:
        seq[1: 1 + len(nbrs)] = nbrs

    patch_nodes = []
    for p in range(num_patches):
        block = seq[p * patch_size: (p + 1) * patch_size]
        patch_nodes.append(block[block > 0])

    return patch_nodes, num_patches


# ---------------------------------------------------------------------------
# Temporal subgraph construction
# ---------------------------------------------------------------------------

def build_temporal_subgraph(all_ids: np.ndarray, t: float,
                             node_nbr_ids, node_nbr_times):
    """
    Build the subgraph among nodes in all_ids (sorted ascending) using edges with timestamp < t.

    Returns:
        src_e, dst_e  : np.ndarray[int64], local edge endpoints in [0, N_eff)
        N_eff         : int, number of nodes with degree > 0
        old_to_new    : np.ndarray[int64], shape (N_prime,), maps all_ids index → N_eff index (-1 if isolated)
    Returns (None, None, None, None) if subgraph has fewer than 2 connected nodes.
    """
    N_prime    = len(all_ids)
    edge_a_buf = []
    edge_c_buf = []

    for uid in all_ids:
        uid = int(uid)
        if uid >= len(node_nbr_ids) or len(node_nbr_ids[uid]) == 0:
            continue
        nbr_ids_u   = node_nbr_ids[uid]
        nbr_times_u = node_nbr_times[uid]
        cutoff = int(np.searchsorted(nbr_times_u, t, side='left'))
        if cutoff == 0:
            continue

        hist_nbrs = nbr_ids_u[:cutoff]
        u_local   = int(np.searchsorted(all_ids, uid))

        idx      = np.searchsorted(all_ids, hist_nbrs)
        idx_clip = np.clip(idx, 0, N_prime - 1)
        in_mask  = (hist_nbrs != uid) & (all_ids[idx_clip] == hist_nbrs)
        if not np.any(in_mask):
            continue

        v_locals = idx_clip[in_mask]
        u_locs   = np.full(len(v_locals), u_local, dtype=np.int64)
        edge_a_buf.append(np.minimum(u_locs, v_locals))
        edge_c_buf.append(np.maximum(u_locs, v_locals))

    if not edge_a_buf:
        return None, None, None, None

    all_a = np.concatenate(edge_a_buf)
    all_c = np.concatenate(edge_c_buf)

    # Deduplicate
    edge_encoded = all_a * (N_prime + 1) + all_c
    _, uniq_idx  = np.unique(edge_encoded, return_index=True)
    dedup_src = all_a[uniq_idx]
    dedup_dst = all_c[uniq_idx]

    # Filter isolated nodes
    deg          = np.bincount(np.concatenate([dedup_src, dedup_dst]), minlength=N_prime)
    core_indices = np.where(deg > 0)[0]
    N_eff        = len(core_indices)
    if N_eff < 2:
        return None, None, None, None

    old_to_new = np.full(N_prime, -1, dtype=np.int64)
    old_to_new[core_indices] = np.arange(N_eff, dtype=np.int64)

    src_e = old_to_new[dedup_src]
    dst_e = old_to_new[dedup_dst]

    return src_e, dst_e, N_eff, old_to_new


# ---------------------------------------------------------------------------
# Per-patch N_eff index lookup
# ---------------------------------------------------------------------------

def get_patch_eff_indices(patch_node_gids: np.ndarray,
                          all_ids: np.ndarray, old_to_new: np.ndarray,
                          N_prime: int) -> np.ndarray:
    """
    Map global node IDs in patch_node_gids to their N_eff-level indices.
    Returns int64 array of valid indices (>= 0 only).
    """
    if len(patch_node_gids) == 0:
        return np.empty(0, dtype=np.int64)
    idx      = np.searchsorted(all_ids, patch_node_gids)
    idx_clip = np.clip(idx, 0, N_prime - 1)
    valid    = all_ids[idx_clip] == patch_node_gids
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)
    local_idx = idx_clip[valid]
    eff_idx   = old_to_new[local_idx]
    return eff_idx[eff_idx >= 0]


# ---------------------------------------------------------------------------
# Vectorised patch-pair aggregation
# ---------------------------------------------------------------------------

def aggregate_patch_pairs(M3d: torch.Tensor, all_patch_eff: list, L: int):
    """
    For each patch pair (pi, pj), compute sum_{u in eff_pi, v in eff_pj} M3d[:, u, v].

    Args:
        M3d           : torch.Tensor, shape (dist_len, N_eff, N_eff)
        all_patch_eff : list[np.ndarray], length L, each array contains N_eff indices
        L             : int, total patches (src_P + dst_P)

    Returns sparse COO arrays:
        rows  : np.ndarray[int16], shape (nnz,)
        cols  : np.ndarray[int16], shape (nnz,)
        vals  : np.ndarray[float16], shape (nnz, dist_len)

    Algorithm (vectorised):
        indicator[pi, u] = 1 if u ∈ eff_pi, else 0          shape (L, N_eff)
        row_agg = einsum('lu,kuv->klv', indicator, M3d)      shape (L, dist_len, N_eff)
        patch_agg = einsum('klv,mv->lmk', row_agg, indicator) → (L, L, dist_len)
        Sparsify: keep entries where patch_agg.any(dim=-1).
    """
    dist_len, N_eff, _ = M3d.shape

    # Build indicator matrix (L, N_eff)
    indicator = torch.zeros(L, N_eff, dtype=torch.float32)
    for pi, eff in enumerate(all_patch_eff):
        if len(eff) > 0:
            indicator[pi, torch.from_numpy(eff).long()] = 1.0

    if indicator.sum() == 0:
        empty16 = np.empty(0, dtype=np.int16)
        return empty16, empty16, np.empty((0, dist_len), dtype=np.float16)

    # row_agg[pi, k, v] = sum_{u in eff_pi} M3d[k, u, v]
    # shape: (L, dist_len, N_eff)
    row_agg = torch.einsum('lu,kuv->lkv', indicator, M3d)

    # patch_agg[pi, pj, k] = sum_{v in eff_pj} row_agg[pi, k, v]
    # shape: (L, L, dist_len)
    patch_agg = torch.einsum('lkv,mv->lmk', row_agg, indicator)

    # Sparsify
    nonzero_mask = patch_agg.abs().sum(dim=-1) > 0   # (L, L)
    pi_idx, pj_idx = nonzero_mask.nonzero(as_tuple=True)

    if pi_idx.numel() == 0:
        empty16 = np.empty(0, dtype=np.int16)
        return empty16, empty16, np.empty((0, dist_len), dtype=np.float16)

    vals_np = torch.log1p(patch_agg[pi_idx, pj_idx, :]).numpy().astype(np.float16)  # (nnz, dist_len)
    rows_np = pi_idx.numpy().astype(np.int16)
    cols_np = pj_idx.numpy().astype(np.int16)

    return rows_np, cols_np, vals_np


# ---------------------------------------------------------------------------
# Process a single interaction
# ---------------------------------------------------------------------------

def process_interaction(src: int, dst: int, t: float,
                        node_nbr_ids, node_nbr_times,
                        max_seq_len: int, patch_size: int,
                        dist_len: int, n_root_nodes_fact: float,
                        min_root_nodes: int, max_bfs_depth: int,
                        max_tries_per_depth: int):
    """
    Compute the precomputed SPSE entry for one (src, dst, t) interaction.

    Returns:
        None   if subgraph has no edges (c_spse stays zero at training time)
        dict   with keys src_P, dst_P, row, col, vals
    """
    # --- 1. Padded sequences → patch node sets ---
    src_patch_nodes, src_P = get_patch_nodes(
        src, t, node_nbr_ids, node_nbr_times, max_seq_len, patch_size)
    dst_patch_nodes, dst_P = get_patch_nodes(
        dst, t, node_nbr_ids, node_nbr_times, max_seq_len, patch_size)

    # --- 2. Union of all patch nodes ---
    all_node_parts = [p for p in src_patch_nodes + dst_patch_nodes if len(p) > 0]
    if not all_node_parts:
        return None
    all_ids  = np.unique(np.concatenate(all_node_parts))
    all_ids  = all_ids[all_ids > 0]
    N_prime  = len(all_ids)
    if N_prime < 2:
        return None

    # --- 3. Build temporal subgraph ---
    src_e, dst_e, N_eff, old_to_new = build_temporal_subgraph(
        all_ids, t, node_nbr_ids, node_nbr_times)
    if src_e is None:
        return None

    # --- 4. Compute SPSE ---
    edge_index_eff = torch.from_numpy(np.stack([src_e, dst_e])).long()
    n_root = max(min_root_nodes, int(N_eff * n_root_nodes_fact))

    try:
        M = get_simple_paths_count(
            edge_index_eff,
            dist_len=dist_len,
            n_root_nodes=n_root,
            max_bfs_depth=max_bfs_depth,
            max_tries_per_depth=max_tries_per_depth,
            n_reps=1,
            directed=False,
            allow_cycles=False,
            max_num_nodes=N_eff,
        )
    except Exception:
        return None

    M3d = M.squeeze(0).float()   # (dist_len, N_eff, N_eff)

    # --- 5. Map each patch to its N_eff indices ---
    L = src_P + dst_P
    all_patch_eff = []
    for p_nodes in src_patch_nodes + dst_patch_nodes:
        all_patch_eff.append(
            get_patch_eff_indices(p_nodes, all_ids, old_to_new, N_prime))

    # --- 6. Vectorised sum-aggregation per patch pair ---
    rows, cols, vals = aggregate_patch_pairs(M3d, all_patch_eff, L)

    if len(rows) == 0:
        return None

    return {
        'src_P': np.int16(src_P),
        'dst_P': np.int16(dst_P),
        'row':   rows,   # np.int16 (nnz,)
        'col':   cols,   # np.int16 (nnz,)
        'vals':  vals,   # np.float16 (nnz, dist_len)
    }


# ---------------------------------------------------------------------------
# Chunk worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _process_chunk_worker(task):
    """
    Process one chunk in a worker process.

    task = (chunk_id, chunk_start, chunk_end,
            src_ids, dst_ids, timestamps, edge_ids, out_dir, overwrite)

    Returns (chunk_id, n_success, n_none, n_error).
    """
    (chunk_id, chunk_start, chunk_end,
     src_ids, dst_ids, timestamps, edge_ids,
     out_dir, overwrite) = task

    chunk_path = os.path.join(out_dir, f'chunk_{chunk_id:05d}.pkl')
    if os.path.exists(chunk_path) and not overwrite:
        print(f'  Chunk {chunk_id:05d} already exists, skipping.', flush=True)
        return chunk_id, 0, 0, 0

    args = _G_args
    n_success = n_none = n_error = 0
    chunk_data = {}
    t_start = time.time()

    for i in range(chunk_start, chunk_end):
        eid = int(edge_ids[i])
        src = int(src_ids[i])
        dst = int(dst_ids[i])
        t   = float(timestamps[i])

        try:
            entry = process_interaction(
                src, dst, t,
                _G_node_nbr_ids, _G_node_nbr_times,
                args.max_input_sequence_length,
                args.patch_size,
                args.dist_len,
                args.n_root_nodes_fact,
                args.min_root_nodes,
                args.max_bfs_depth,
                args.max_tries_per_depth,
            )
            chunk_data[eid] = entry
            if entry is None:
                n_none += 1
            else:
                n_success += 1
        except Exception as exc:
            chunk_data[eid] = None
            n_error += 1
            if args.verbose:
                print(f'    [ERROR] chunk={chunk_id} i={i} eid={eid}: {exc}', flush=True)

    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk_data, f, protocol=4)

    elapsed = time.time() - t_start
    size    = chunk_end - chunk_start
    rate    = size / elapsed if elapsed > 0 else 0.0
    print(f'  → Chunk {chunk_id:05d} done: {size} interactions  '
          f'ok={n_success} none={n_none} err={n_error}  '
          f'rate={rate:.1f}/s', flush=True)
    return chunk_id, n_success, n_none, n_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    t_wall = time.time()

    # ── Load CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join('processed_data', args.dataset_name,
                            f'ml_{args.dataset_name}.csv')
    print(f'Loading {csv_path} ...')
    graph_df = pd.read_csv(csv_path)

    src_ids    = graph_df.u.values.astype(np.int64)
    dst_ids    = graph_df.i.values.astype(np.int64)
    timestamps = graph_df.ts.values.astype(np.float64)
    edge_ids   = graph_df.idx.values.astype(np.int64)
    N          = len(src_ids)
    print(f'  {N} interactions, nodes up to {max(src_ids.max(), dst_ids.max())}')

    # ── Build neighbor structure (full data, bidirectional) ─────────────────
    print('Building neighbor structure ...')
    node_nbr_ids, node_nbr_times = build_neighbor_structure(src_ids, dst_ids, timestamps)
    print(f'  Done. {len(node_nbr_ids)} node slots.')

    # ── Output directory ────────────────────────────────────────────────────
    tag     = f'ps{args.patch_size}_dl{args.dist_len}'
    out_dir = os.path.join('processed_data', args.dataset_name,
                           f'spse_precomputed_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output → {out_dir}')

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        'dataset_name':            args.dataset_name,
        'patch_size':              args.patch_size,
        'max_input_sequence_length': args.max_input_sequence_length,
        'dist_len':                args.dist_len,
        'n_root_nodes_fact':       args.n_root_nodes_fact,
        'min_root_nodes':          args.min_root_nodes,
        'max_bfs_depth':           args.max_bfs_depth,
        'max_tries_per_depth':     args.max_tries_per_depth,
        'aggregation':             'sum',
        'chunk_size':              args.chunk_size,
        'num_interactions':        N,
        'num_chunks':              math.ceil(N / args.chunk_size),
        # edge_id_to_chunk: edge_ids[i] lives in chunk i // chunk_size
        # Loading tip: load all chunk_*.pkl into one dict at training time.
    }
    meta_path = os.path.join(out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Process by chunk (parallel) ──────────────────────────────────────────
    num_chunks   = math.ceil(N / args.chunk_size)
    num_workers  = min(args.num_workers, num_chunks)
    n_success = n_none = n_error = 0

    tasks = []
    for chunk_id in range(num_chunks):
        chunk_start = chunk_id * args.chunk_size
        chunk_end   = min(chunk_start + args.chunk_size, N)
        tasks.append((
            chunk_id, chunk_start, chunk_end,
            src_ids, dst_ids, timestamps, edge_ids,
            out_dir, args.overwrite,
        ))

    print(f'Processing {num_chunks} chunks with {num_workers} workers ...')

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(node_nbr_ids, node_nbr_times, args),
    ) as pool:
        for _, s, nn, e in pool.imap_unordered(_process_chunk_worker, tasks):
            n_success += s
            n_none    += nn
            n_error   += e

    elapsed_total = time.time() - t_wall
    print(f'\nFinished. total={N} success={n_success} none={n_none} error={n_error}')
    print(f'Total time: {elapsed_total/60:.1f} min  |  Output: {out_dir}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute SPSE for dynamic graph datasets.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (folder under processed_data/)')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size — must match training config (default: 4)')
    parser.add_argument('--max_input_sequence_length', type=int, default=512,
                        help='Max sequence length — must match training config (default: 512)')
    parser.add_argument('--dist_len', type=int, default=16,
                        help='SPSE max path length K (default: 16)')
    parser.add_argument('--n_root_nodes_fact', type=float, default=0.4,
                        help='Fraction of subgraph nodes used as SPSE root R (default: 0.4)')
    parser.add_argument('--min_root_nodes', type=int, default=4,
                        help='Minimum number of root nodes (default: 4)')
    parser.add_argument('--max_bfs_depth', type=int, default=2,
                        help='Max DFS depth before BFS switch DDFS (default: 2)')
    parser.add_argument('--max_tries_per_depth', type=int, default=7,
                        help='Trials per depth N (default: 7)')
    parser.add_argument('--chunk_size', type=int, default=5000,
                        help='Interactions per chunk file (default: 5000)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel worker processes (default: 8)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing chunk files (disables resume)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-interaction error messages')
    args = parser.parse_args()
    main(args)
