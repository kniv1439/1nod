# -*- coding: utf-8 -*-
import sys
sys.argv = ['test', '--dataset_name', 'CanParl', '--model_name', 'DyGFormerF', '--load_best_configs', '--num_runs', '3', '--num_epochs', '100', '--num_heads', '2', '--num_kv_heads', '1']

from utils.load_configs import get_link_prediction_args

args = get_link_prediction_args(is_evaluation=False)
print(f"num_heads: {args.num_heads}")
print(f"num_kv_heads: {args.num_kv_heads}")
print("Test passed!")
