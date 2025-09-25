#!/bin/bash

cd /cis/home/xhan56/code/dami/src

python train_pamap_multimodal_baseline.py \
    --subject_id 1 \
    --seq_len 100 \
    --num_moe_layers 3 \
    --cuda_device 2 \
    --use_wandb \
    --moe_num_experts 8 \
    --moe_k 2 \
    --moe_expert_hidden_dim 128 \
    --seed 42 \
    --plot_expert_activations

python train_pamap_multimodal_baseline.py \
    --subject_id 2 \
    --seq_len 100 \
    --num_moe_layers 3 \
    --cuda_device 2 \
    --use_wandb \
    --moe_num_experts 8 \
    --moe_k 2 \
    --moe_expert_hidden_dim 128 \
    --seed 42 \
    --plot_expert_activations