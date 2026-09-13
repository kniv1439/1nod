#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看各模型在 wikipedia / reddit 数据集下已保存的链路预测（LP）模型参数配置。
扫描 saved_models/<model_name>/<dataset>/ 下的 .pkl checkpoint，
从 state_dict 推断结构相关参数；若同目录存在 train_config.json 则一并读取并展示。
在项目根目录运行: python inspect_wikipedia_lp_saved_configs.py
"""
import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch

SAVED_BASE = os.path.join(_script_dir, "saved_models")
DATASETS = ("wikipedia", "reddit")
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

    # 1) channel_embedding_dim
    for key in state_dict:
        k = _strip_module(key)
        if "projection_layer" in k and ("node" in k or "edge" in k) and "weight" in k:
            shp = state_dict[key].shape
            if len(shp) == 2:
                cfg["channel_embedding_dim"] = int(shp[0])
                break
    if "channel_embedding_dim" not in cfg:
        for key in state_dict:
            k = _strip_module(key)
            if "output_layer" in k and "weight" in k:
                shp = state_dict[key].shape
                if len(shp) == 2 and shp[1] % 4 == 0:
                    cfg["channel_embedding_dim"] = int(shp[1]) // 4
                    break

    # 2) num_layers
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

    # 3) num_heads
    if "channel_embedding_dim" in cfg and "num_heads" not in cfg:
        attn_dim = 4 * cfg["channel_embedding_dim"]
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

    # 4) time_feat_dim
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
                    in_dim = int(shp[1])
                    for ps in (1, 2, 4):
                        if in_dim % ps == 0:
                            cfg["time_feat_dim"] = in_dim // ps
                            break
                break

    # 5) patch_size（wikipedia 常用 position_feat_dim=172 等）
    if "patch_size" not in cfg:
        for key in state_dict:
            k = _strip_module(key)
            if "projection_layer" in k and "node" in k and "weight" in k:
                in_dim = int(state_dict[key].shape[1])
                for ps in (4, 2, 1):
                    if in_dim % ps == 0 and (in_dim // ps) in (172, 100, 64, 688):
                        cfg["patch_size"] = ps
                        break
                break

    return cfg


def _load_train_config(folder_path, save_name):
    """若存在 train_config.json 则读取并返回其中 KEY_PARAMS 相关字段。"""
    path = os.path.join(folder_path, "train_config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: data.get(k) for k in KEY_PARAMS if k in data}
    except Exception:
        return None


def _find_saved_models(dataset, link_prediction_only=True):
    """返回 [(model_name, save_name, pkl_path, folder_path), ...]。仅保留链路预测模型（排除 node_classification_*）。"""
    out = []
    if not os.path.isdir(SAVED_BASE):
        return out
    for model_name in sorted(os.listdir(SAVED_BASE)):
        dataset_dir = os.path.join(SAVED_BASE, model_name, dataset)
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
                out.append((model_name, save_name, pkl_path, folder))
    return out


def main():
    key_params = KEY_PARAMS
    any_found = False
    for dataset in DATASETS:
        rows = _find_saved_models(dataset)
        if not rows:
            print(f"\n{dataset.upper()} 未找到已保存链路预测模型 (.pkl)。")
            continue
        any_found = True
        print(f"\n{dataset.upper()} 已保存链路预测（LP）模型参数配置（共 {len(rows)} 个 checkpoint）")
        print("-" * 110)
        for model_name, save_name, pkl_path, folder in rows:
            cfg = {}
            train_cfg = _load_train_config(folder, save_name)
            if train_cfg:
                cfg.update(train_cfg)
            try:
                try:
                    state_dict = torch.load(pkl_path, map_location="cpu", weights_only=True)
                except TypeError:
                    state_dict = torch.load(pkl_path, map_location="cpu")
                inferred = _infer_from_state_dict(state_dict)
                for k, v in inferred.items():
                    if v is not None:
                        cfg[k] = v
            except Exception as e:
                print(f"  {model_name:24}  {save_name:36}  [.pkl 加载失败: {e}]")
                continue
            parts = [f"{k}={cfg.get(k, '-')}" for k in key_params]
            print(f"  {model_name:24}  {save_name:36}  " + ", ".join(parts))
        print("-" * 110)
    if not any_found:
        print(f"未在 {SAVED_BASE} 下找到 wikipedia / reddit 任一数据集的已保存链路预测模型 (.pkl)。")
        print("请先做链路预测训练并保存模型。")
    else:
        print("说明: 若目录存在 train_config.json 则优先采用其中参数，其余从 state_dict 推断，未推断出的显示为 '-'。")


if __name__ == "__main__":
    main()
