#!/bin/bash
# 测试 GQA 参数
cd /home/zhangjinhao/nod
CUDA_VISIBLE_DEVICES=1 python train_link_prediction.py \
  --dataset_name CanParl \
  --model_name DyGFormerF \
  --load_best_configs \
  --num_runs 3 \
  --num_epochs 100 \
  --num_heads 2 \
  --num_kv_heads 1
