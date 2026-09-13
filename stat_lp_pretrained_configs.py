#!/usr/bin/env python3
"""
Statistics of all models' link prediction (LP) pretrained configs.
Run this script to dump LP structural configs to configs/lp_pretrained_configs.json
and print a summary table. Used for reference when node classification loads LP backbones.
"""
import json
import os
import sys

# allow importing from project root (run from repo root: python stat_lp_pretrained_configs.py)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from utils.load_configs import get_all_lp_pretrained_configs, get_lp_pretrained_config, LP_STRUCTURAL_KEYS


def main():
    out_dir = "configs"
    os.makedirs(out_dir, exist_ok=True)

    # collect for both datasets (DeformableDyG etc. may differ by dataset)
    result = {
        "wikipedia": get_all_lp_pretrained_configs("wikipedia"),
        "reddit": get_all_lp_pretrained_configs("reddit"),
    }

    out_path = os.path.join(out_dir, "lp_pretrained_configs.json")

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(x) for x in obj]
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(result), f, indent=2, ensure_ascii=False)
    print("Saved configs to", out_path)

    # print summary table (wikipedia and reddit; key structural params)
    key_params = ["channel_embedding_dim", "time_feat_dim", "num_layers", "num_heads", "patch_size", "max_input_sequence_length"]
    for dataset in ["wikipedia", "reddit"]:
        print("\nLP pretrained config summary ({}):".format(dataset))
        print("-" * 80)
        for model_name, cfg in sorted(result[dataset].items()):
            if not cfg:
                print("  {}: (no config / error)".format(model_name))
                continue
            parts = ["{}={}".format(k, cfg.get(k, "-")) for k in key_params if k in cfg]
            print("  {}: ".format(model_name) + ", ".join(parts))
        print("-" * 80)


if __name__ == "__main__":
    main()
