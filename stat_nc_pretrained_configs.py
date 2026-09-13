#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计所有模型在节点分类（NC）之前使用的自主训练好的预训练模型（即 LP 链路预测 checkpoint）的参数配置。
扫描 saved_models/<model_name>/wikipedia 与 saved_models/<model_name>/reddit 下的 .pkl（仅 LP，排除 node_classification_*），
从 state_dict 或 train_config.json 得到结构参数，汇总写入 configs/nc_pretrained_configs.json 并打印表格。
在项目根目录运行: python stat_nc_pretrained_configs.py
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
    """从 state_dict 的 key 和 shape 推断结构参数。"""
    cfg = {}
    if not state_dict:
        return cfg

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


def _load_train_config(folder_path):
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
    """返回 [(model_name, save_name, pkl_path, folder_path), ...]。仅保留 LP（排除 node_classification_*）。"""
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


def _get_config_for_checkpoint(pkl_path, folder_path):
    """对单个 checkpoint 得到参数字典（优先 train_config.json，再 state_dict 推断）。"""
    cfg = {}
    train_cfg = _load_train_config(folder_path)
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
    except Exception:
        pass
    return cfg


def main():
    os.makedirs("configs", exist_ok=True)
    # result[dataset][model_name][save_name] = config dict
    result = {ds: {} for ds in DATASETS}

    for dataset in DATASETS:
        rows = _find_saved_models(dataset)
        for model_name, save_name, pkl_path, folder in rows:
            cfg = _get_config_for_checkpoint(pkl_path, folder)
            if model_name not in result[dataset]:
                result[dataset][model_name] = {}
            result[dataset][model_name][save_name] = cfg

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(x) for x in obj]
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    out_path = os.path.join("configs", "nc_pretrained_configs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(result), f, indent=2, ensure_ascii=False)
    print("已写入:", out_path)

    # 打印汇总表：按数据集 -> 模型 -> 各 checkpoint
    key_params = KEY_PARAMS
    for dataset in DATASETS:
        models_cfgs = result[dataset]
        if not models_cfgs:
            print(f"\n{dataset.upper()} 未找到节点分类可用的 LP 预训练 checkpoint。")
            continue
        total = sum(len(saves) for saves in models_cfgs.values())
        print(f"\n【{dataset.upper()}】节点分类所用 LP 预训练模型参数配置（共 {total} 个 checkpoint）")
        print("-" * 120)
        for model_name in sorted(models_cfgs.keys()):
            for save_name, cfg in sorted(models_cfgs[model_name].items()):
                parts = [f"{k}={cfg.get(k, '-')}" for k in key_params]
                print(f"  {model_name:22}  {save_name:36}  " + ", ".join(parts))
        print("-" * 120)
    print("\n说明: 以上为节点分类前自主训练好的 LP backbone 的配置（来自 train_config.json 或 state_dict 推断）。")


if __name__ == "__main__":
    main()
