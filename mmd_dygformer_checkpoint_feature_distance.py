# -*- coding: utf-8 -*-
"""
Compute train/test MMD distance from trained DyGFormer checkpoint features.

Data scope: interactions whose timestamps fall in each dataset's train or test
window (from ``mmd_dataset_comparison_mmd2.DATASETS``). By default **all** such
edges are used (``--max_interactions 0``). Optional subsampling is only for
speed / memory.

Feature extraction: for every interaction (src, dst, t) in order,
``DyGFormer.compute_src_dst_node_temporal_embeddings`` runs on the full graph
neighbor sampler (history before t), yielding node embeddings; then
``pair_feature = concat(src_emb, dst_emb)`` (batched by ``--batch_size``).

MMD uses blocked RBF-kernel estimates (``--mmd_block_size``), not a dense
n×n Gram matrix.

DyGFormer hyperparameters follow ``utils.load_configs.load_link_prediction_best_configs``
for ``model_name`` DyGFormer. Default checkpoint seed is 1 (e.g. mooc has no seed0).

Checkpoints are loaded from:
  saved_models/DyGFormer/{dataset}/DyGFormer_seed{seed}/DyGFormer_seed{seed}.pkl
or, as fallback:
  saved_models/DyGFormer/{dataset}/DyGFormer_seed{seed}/model.pth
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from mmd_dataset_comparison_mmd2 import DATASETS
from mmd_dygformer_interpretable_raw_stats import (
    TARGET_DATASETS,
    build_data_from_df,
    compute_mmd,
    load_raw_features,
    sample_data,
)
from models.DyGFormer import DyGFormer
from utils.load_configs import load_link_prediction_best_configs
from utils.utils import get_neighbor_sampler


def checkpoint_candidates(dataset_name: str, checkpoint_seed: int) -> Iterable[str]:
    folder = f"saved_models/DyGFormer/{dataset_name}/DyGFormer_seed{checkpoint_seed}"
    yield os.path.join(folder, f"DyGFormer_seed{checkpoint_seed}.pkl")
    yield os.path.join(folder, "model.pth")


def find_checkpoint(dataset_name: str, checkpoint_seed: int) -> str:
    for path in checkpoint_candidates(dataset_name, checkpoint_seed):
        if os.path.exists(path):
            return path
    candidates = "\n".join(checkpoint_candidates(dataset_name, checkpoint_seed))
    raise FileNotFoundError(f"No DyGFormer checkpoint found for {dataset_name}. Tried:\n{candidates}")


def unwrap_checkpoint_state(state) -> Dict[str, torch.Tensor]:
    """Handle plain state_dicts and checkpoint dict wrappers."""
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)}")
    return state


def normalize_state_key(key: str) -> Optional[str]:
    """
    Training usually saves nn.Sequential(DyGFormer, link_predictor), where the
    DyGFormer backbone keys start with "0.". Strip that prefix and skip the
    predictor keys starting with "1.".
    """
    if key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("0."):
        return key[2:]
    if key.startswith("1."):
        return None
    return key


def load_dygformer_checkpoint(model: DyGFormer, dataset_name: str, checkpoint_seed: int, device: str) -> str:
    ckpt_path = find_checkpoint(dataset_name, checkpoint_seed)
    raw_state = torch.load(ckpt_path, map_location=device)
    state = unwrap_checkpoint_state(raw_state)

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []
    skipped_unknown = []

    for key, value in state.items():
        normalized_key = normalize_state_key(key)
        if normalized_key is None:
            continue
        if normalized_key not in model_state:
            skipped_unknown.append(normalized_key)
            continue
        if model_state[normalized_key].shape != value.shape:
            skipped_shape.append((normalized_key, tuple(value.shape), tuple(model_state[normalized_key].shape)))
            continue
        filtered_state[normalized_key] = value

    if not filtered_state:
        raise RuntimeError(
            f"No compatible DyGFormer weights loaded from {ckpt_path}. "
            "Please check model hyperparameters."
        )

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(
        f"Loaded {len(filtered_state)} tensors from {ckpt_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)}, "
        f"shape_skipped={len(skipped_shape)}, unknown_skipped={len(skipped_unknown)})"
    )
    if skipped_shape:
        print("First shape-skipped keys:")
        for item in skipped_shape[:5]:
            print(f"  {item[0]}: checkpoint{item[1]} vs model{item[2]}")
    return ckpt_path


@torch.no_grad()
def extract_dygformer_pair_features(model: DyGFormer, data, batch_size: int) -> np.ndarray:
    features = []

    for start in range(0, data.num_interactions, batch_size):
        end = min(start + batch_size, data.num_interactions)
        src = data.src_node_ids[start:end]
        dst = data.dst_node_ids[start:end]
        ts = data.node_interact_times[start:end]

        src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src,
            dst_node_ids=dst,
            node_interact_times=ts,
        )
        features.append(torch.cat([src_emb, dst_emb], dim=1).cpu())

    return torch.cat(features, dim=0).numpy()


def build_model(name: str, full_data, args: argparse.Namespace) -> DyGFormer:
    node_raw, edge_raw = load_raw_features(name)
    sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=args.seed,
    )

    model = DyGFormer(
        node_raw_features=node_raw,
        edge_raw_features=edge_raw,
        neighbor_sampler=sampler,
        time_feat_dim=args.time_feat_dim,
        channel_embedding_dim=args.channel_embedding_dim,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_input_sequence_length=args.max_input_sequence_length,
        device=args.device,
    )
    return model.to(args.device)


def analyze_dataset(name: str, cfg: dict, args: argparse.Namespace) -> dict:
    df = pd.read_csv(cfg["path"])
    full_data = build_data_from_df(df)

    train_df = df[(df["ts"] >= cfg["train_ts_start"]) & (df["ts"] <= cfg["train_ts_end"])]
    test_df = df[(df["ts"] >= cfg["test_ts_start"]) & (df["ts"] <= cfg["test_ts_end"])]

    train_data_full = build_data_from_df(train_df)
    test_data_full = build_data_from_df(test_df)
    train_data = sample_data(train_data_full, args.max_interactions, args.seed)
    test_data = sample_data(test_data_full, args.max_interactions, args.seed + 1)

    model = build_model(name, full_data, args)
    ckpt_path = load_dygformer_checkpoint(model, name, args.checkpoint_seed, args.device)
    model.eval()

    train_feat = extract_dygformer_pair_features(model, train_data, args.batch_size)
    test_feat = extract_dygformer_pair_features(model, test_data, args.batch_size)
    mmd2, dist = compute_mmd(train_feat, test_feat, args.device, args.mmd_block_size)

    print(
        f"[{name:12s}] used={train_data.num_interactions}/{test_data.num_interactions} "
        f"MMD^2={mmd2:.8f} MMD_distance={dist:.8f}"
    )

    return {
        "dataset": name,
        "type": cfg["type"],
        "checkpoint_path": ckpt_path,
        "train_interactions_full": train_data_full.num_interactions,
        "test_interactions_full": test_data_full.num_interactions,
        "train_interactions_used": train_data.num_interactions,
        "test_interactions_used": test_data.num_interactions,
        "MMD_dygformer_emb_squared": mmd2,
        "MMD_dygformer_emb_distance": dist,
    }


def parse_dataset_filter(value: str) -> Optional[set]:
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/test MMD distance using trained DyGFormer checkpoint embeddings."
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Neighbor-sampler seed (and subsampling seed if --max_interactions > 0).",
    )
    parser.add_argument(
        "--checkpoint_seed",
        type=int,
        default=1,
        help="Seed suffix used by saved checkpoint folder (default 1; mooc has no seed0).",
    )
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset names, or all.")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument(
        "--max_interactions",
        type=int,
        default=0,
        help="Cap train/test edges per split after time-window filter (0 = use all edges in each window).",
    )
    parser.add_argument("--mmd_block_size", type=int, default=512)

    parser.add_argument("--sample_neighbor_strategy", type=str, default="recent")
    parser.add_argument("--time_scaling_factor", type=float, default=0.0)
    parser.add_argument(
        "--time_feat_dim",
        type=int,
        default=100,
        help="Time embedding dim (not overridden by load_link_prediction_best_configs for DyGFormer).",
    )
    parser.add_argument(
        "--max_input_sequence_length",
        type=int,
        default=512,
        help="Max input sequence length per node (aligns with train_link_prediction defaults).",
    )
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    args.model_name = "DyGFormer"
    load_link_prediction_best_configs(args)

    dataset_filter = parse_dataset_filter(args.datasets)
    selected = {
        name: cfg
        for name, cfg in DATASETS.items()
        if name in TARGET_DATASETS and (dataset_filter is None or name in dataset_filter)
    }

    rows = []
    for name, cfg in selected.items():
        try:
            rows.append(analyze_dataset(name, cfg, args))
        except Exception as exc:
            print(f"[{name}] failed: {exc}")
            rows.append(
                {
                    "dataset": name,
                    "type": cfg.get("type", "unknown"),
                    "error": str(exc),
                    "MMD_dygformer_emb_squared": float("nan"),
                    "MMD_dygformer_emb_distance": float("nan"),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        "MMD_dygformer_emb_distance", ascending=False, na_position="last"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mmd_dygformer_checkpoint_feature_distance.csv")
    result.to_csv(out_path, index=False)

    print("\nSummary:")
    print(result.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
