#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看各模型在 reddit 数据集下已保存模型的参数配置。
扫描 saved_models/<model_name>/reddit/ 下的 .pkl checkpoint，
从 state_dict 推断结构相关参数并打印汇总表。
在项目根目录运行: python inspect_reddit_lp_saved_configs.py
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch

SAVED_BASE = os.path.join(_script_dir, "saved_models")
DATASET = "reddit"
# 希望汇总的结构参数（能从 state_dict 推断的尽量推断，否则标 -）
KEY_PARAMS = [
    "channel_embedding_dim",
    "time_feat_dim",
    "num_layers",
    "num_heads",
    "patch_size",
    "max_input_sequence_length",
]


def _strip_module(key):
    """DataParallel 会加 'module.' 前缀"""
    return key.replace("module.", "") if key.startswith("module.") else key


def _infer_from_state_dict(state_dict):
    """从 state_dict 的 key 和 shape 推断结构参数。PyTorch Linear 的 weight 形状为 (out_features, in_features)。"""
    cfg = {}
    if not state_dict:
        return cfg

    # 1) channel_embedding_dim：DyGFormer/DIFFDyG 等用 projection_layer.node.weight，Linear(out=channel_embedding_dim, in=patch*feat_dim)
    #    weight.shape = (channel_embedding_dim, in)，故取 shape[0]
    for key in state_dict:
        k = _strip_module(key)
        if "projection_layer" in k and ("node" in k or "edge" in k) and "weight" in k:
            shp = state_dict[key].shape
            if len(shp) == 2:
                cfg["channel_embedding_dim"] = int(shp[0])
                break
    if "channel_embedding_dim" not in cfg:
        # 备选：output_layer.weight shape (node_feat_dim, num_channels*channel_embedding_dim)，num_channels 多为 4
        for key in state_dict:
            k = _strip_module(key)
            if "output_layer" in k and "weight" in k:
                shp = state_dict[key].shape
                if len(shp) == 2 and shp[1] % 4 == 0:
                    cfg["channel_embedding_dim"] = int(shp[1]) // 4
                    break

    # 2) num_layers：transformers.0.xxx, transformers.1.xxx 或 layers.0 等
    for key in state_dict:
        k = _strip_module(key)
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "transformers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                cfg["num_layers"] = max(cfg.get("num_layers", 0), idx + 1)
                break
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                cfg["num_layers"] = max(cfg.get("num_layers", 0), idx + 1)
                break

    # 3) num_heads：从 transformer 里 in_proj_weight / attention 的维度推 (attention_dim 能被 num_heads 整除)
    if "channel_embedding_dim" in cfg and "num_heads" not in cfg:
        attn_dim = 4 * cfg["channel_embedding_dim"]  # num_channels=4
        for key in state_dict:
            k = _strip_module(key)
            if ("in_proj_weight" in k or "attn" in k) and "weight" in k:
                shp = state_dict[key].shape
                if len(shp) >= 2:
                    d = int(shp[0]) if shp[0] == attn_dim else int(shp[-1])
                    if d == attn_dim:
                        for h in (2, 4, 8):
                            if attn_dim % h == 0 and (attn_dim // h) % 2 == 0:
                                cfg["num_heads"] = h
                                break
                    if cfg.get("num_heads"):
                        break

    # 4) time_feat_dim：TimeEncoder 的 w 为 Linear(1, time_dim)，weight.shape = (time_dim, 1)，取 shape[0]
    for key in state_dict:
        k = _strip_module(key)
        if "time_encoder" in k and "weight" in k:
            shp = state_dict[key].shape
            if len(shp) >= 1:
                cfg["time_feat_dim"] = int(shp[0])
                break
    if "time_feat_dim" not in cfg:
        for key in state_dict:
            k = _strip_module(key)
            if "projection_layer" in k and "time" in k and "weight" in k:
                shp = state_dict[key].shape
                if len(shp) == 2 and cfg.get("channel_embedding_dim"):
                    # in_features = patch_size * time_feat_dim，常见 patch_size=4
                    in_dim = int(shp[1])
                    for ps in (1, 2, 4):
                        if in_dim % ps == 0:
                            cfg["time_feat_dim"] = in_dim // ps
                            break
                break

    # 5) patch_size：projection_layer.node.weight shape[1] = patch_size * node_feat_dim；reddit 常用 node_feat_dim=172
    if "patch_size" not in cfg:
        for key in state_dict:
            k = _strip_module(key)
            if "projection_layer" in k and "node" in k and "weight" in k:
                in_dim = int(state_dict[key].shape[1])
                for ps in (4, 2, 1):
                    if in_dim % ps == 0 and (in_dim // ps) in (172, 100, 64):
                        cfg["patch_size"] = ps
                        break
                break

    return cfg


def _find_reddit_saved_models(link_prediction_only: bool = True):
    """返回 [(model_name, save_name, pkl_path), ...]。link_prediction_only=True 时只保留链路预测模型（排除 node_classification_*）。"""
    out = []
    if not os.path.isdir(SAVED_BASE):
        return out
    for model_name in sorted(os.listdir(SAVED_BASE)):
        dataset_dir = os.path.join(SAVED_BASE, model_name, DATASET)
        if not os.path.isdir(dataset_dir):
            continue
        for save_name in sorted(os.listdir(dataset_dir)):
            if link_prediction_only and "node_classification" in save_name:
                continue
            folder = os.path.join(dataset_dir, save_name)
            if not os.path.isdir(folder):
                continue
            pkl_path = os.path.join(folder, f"{save_name}.pkl")
            if os.path.isfile(pkl_path):
                out.append((model_name, save_name, pkl_path))
    return out


def main():
    rows = _find_reddit_saved_models()
    if not rows:
        print(f"未在 {SAVED_BASE} 下找到 reddit 数据集的已保存模型 (.pkl)。")
        print("请先对 reddit 做链路预测训练并保存模型。")
        return

    print(f"Reddit 已保存模型参数配置（仅链路预测 LP，共 {len(rows)} 个 checkpoint）")
    print("-" * 100)
    key_params = KEY_PARAMS

    for model_name, save_name, pkl_path in rows:
        try:
            try:
                state_dict = torch.load(pkl_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(pkl_path, map_location="cpu")
        except Exception as e:
            print(f"  {model_name} / {save_name}: 加载失败 - {e}")
            continue
        cfg = _infer_from_state_dict(state_dict)
        parts = [f"{k}={cfg.get(k, '-')}" for k in key_params]
        print(f"  {model_name:24}  {save_name:32}  " + ", ".join(parts))
    print("-" * 100)
    print("说明: 上述参数从 state_dict 的 key/shape 推断，未推断出的显示为 '-'。")


if __name__ == "__main__":
    main()
