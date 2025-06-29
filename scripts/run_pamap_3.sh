#!/bin/bash

# python ../src/train_pamap.py --subject_id 1 --seq_len 50 --num_moe_layers 1 --cuda_device 2 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 50 --num_moe_layers 2 --cuda_device 2 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 50 --num_moe_layers 3 --cuda_device 2 --use_wandb

# python ../src/train_pamap.py --subject_id 5 --seq_len 100 --num_moe_layers 3 --cuda_device 2 --use_wandb --rus_dominance_threshold 0.4

# python ../src/train_pamap.py --subject_id 6 --seq_len 100 --num_moe_layers 3 --cuda_device 2 --use_wandb --rus_dominance_threshold 0.4

python ../src/train_pamap_baseline.py \
    --subject_id 2 \
    --seq_len 100 \
    --num_moe_layers 3 \
    --cuda_device 2 \
    --use_wandb \
    --moe_num_experts 16 \
    --moe_k 2 \
    --moe_expert_hidden_dim 128 \
    --seed 42 \
    --lambda_load 0.1 \
    --plot_expert_activations \
    --plot_num_samples 64