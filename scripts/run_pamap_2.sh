#!/bin/bash

# python ../src/train_pamap.py --subject_id 1 --seq_len 75 --num_moe_layers 1 --cuda_device 1 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 75 --num_moe_layers 2 --cuda_device 1 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 75 --num_moe_layers 3 --cuda_device 1 --use_wandb

# python ../src/train_pamap.py --subject_id 3 --seq_len 100 --num_moe_layers 3 --cuda_device 1 --use_wandb --rus_dominance_threshold 0.4

# python ../src/train_pamap.py --subject_id 4 --seq_len 100 --num_moe_layers 3 --cuda_device 1 --use_wandb --rus_dominance_threshold 0.4

# python ../src/pamap_rus.py \
#     --subject_id 2 \
#     --max_lag 10 \
#     --bins 8 \
#     --dominance_threshold 0.4 \
#     --dominance_percentage 0.9 

# Loop through different threshold values for threshold_u, threshold_r, and threshold_s
for threshold_u in 0.3 0.4 0.5; do
    for threshold_r in 0.05 0.075 0.1; do
        for threshold_s in 0.05 0.075 0.1; do
            echo "Running with thresholds: U=$threshold_u, R=$threshold_r, S=$threshold_s"
            python ../src/train_pamap.py \
                --subject_id 2 \
                --seq_len 10 \
                --num_moe_layers 3 \
                --cuda_device 1 \
                --use_wandb \
                --rus_max_lag 10 \
                --rus_bins 8 \
                --moe_num_experts 16 \
                --moe_num_synergy_experts 2 \
                --moe_k 2 \
                --moe_expert_hidden_dim 128 \
                --moe_router_gru_hidden_dim 64 \
                --moe_router_token_processed_dim 64 \
                --moe_router_attn_key_dim 32 \
                --moe_router_attn_value_dim 32 \
                --seed 42 \
                --threshold_u $threshold_u \
                --threshold_r $threshold_r \
                --threshold_s $threshold_s \
                --lambda_u 0.1 \
                --lambda_r 0.1 \
                --lambda_s 0.1 \
                --lambda_load 0.1
        done
    done
done
