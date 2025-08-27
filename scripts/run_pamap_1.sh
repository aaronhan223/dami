#!/bin/bash

# python ../src/train_pamap.py --subject_id 1 --seq_len 100 --num_moe_layers 1 --cuda_device 0 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 100 --num_moe_layers 2 --cuda_device 0 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 100 --num_moe_layers 3 --cuda_device 0 --use_wandb --rus_dominance_threshold 0.4

# python ../src/train_pamap.py --subject_id 2 --seq_len 100 --num_moe_layers 3 --cuda_device 0 --use_wandb --rus_dominance_threshold 0.4

# python ../src/pamap_rus.py \
#     --subject_id 1 \
#     --max_lag 10 \
#     --bins 8 \
#     --dominance_threshold 0.4 \
#     --dominance_percentage 0.9 

# Loop through different threshold values for threshold_u, threshold_r, and threshold_s
# for threshold_u in 0.3 0.4 0.5; do
#     for threshold_r in 0.05 0.075 0.1; do
#         for threshold_s in 0.05 0.075 0.1; do
#             echo "Running with thresholds: U=$threshold_u, R=$threshold_r, S=$threshold_s"
#             python ../src/train_pamap.py \
#                 --subject_id 1 \
#                 --seq_len 10 \
#                 --num_moe_layers 3 \
#                 --cuda_device 0 \
#                 --use_wandb \
#                 --rus_max_lag 10 \
#                 --rus_bins 8 \
#                 --moe_num_experts 16 \
#                 --moe_num_synergy_experts 2 \
#                 --moe_k 2 \
#                 --moe_expert_hidden_dim 128 \
#                 --moe_router_gru_hidden_dim 64 \
#                 --moe_router_token_processed_dim 64 \
#                 --moe_router_attn_key_dim 32 \
#                 --moe_router_attn_value_dim 32 \
#                 --seed 42 \
#                 --threshold_u $threshold_u \
#                 --threshold_r $threshold_r \
#                 --threshold_s $threshold_s \
#                 --lambda_u 0.1 \
#                 --lambda_r 0.1 \
#                 --lambda_s 0.1 \
#                 --lambda_load 0.1
#         done
#     done
# done

# python ../src/pamap_rus_multimodal.py \
#     --method batch \
#     --subject_id 1 \
#     --max_lag 10 \
#     --dominance_threshold 0.4 \
#     --dominance_percentage 0.9 \
#     --gpu 0 \
#     --hidden_dim 64 \
#     --layers 3 \
#     --lr 0.001 \
#     --discrim_epochs 30 \
#     --ce_epochs 15 \
#     --activation relu \
#     --embed_dim 20 \
#     --batch_size 512 \
#     --n_batches 3 \
#     --seed 42

python ../src/train_pamap_multimodal.py \
    --subject_id 1 \
    --seq_len 100 \
    --window_step 50 \
    --val_split 0.2 \
    --rus_max_lag 10 \
    --rus_bins 4 \
    --d_model 128 \
    --nhead 4 \
    --d_ff 256 \
    --num_encoder_layers 6 \
    --num_moe_layers 3 \
    --dropout 0.1 \
    --modality_encoder_layers 2 \
    --moe_num_experts 8 \
    --moe_num_synergy_experts 2 \
    --moe_k 2 \
    --moe_expert_hidden_dim 128 \
    --moe_capacity_factor 1.25 \
    --moe_router_gru_hidden_dim 64 \
    --moe_router_token_processed_dim 64 \
    --moe_router_attn_key_dim 32 \
    --moe_router_attn_value_dim 32 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --clip_grad_norm 1.0 \
    --use_lr_scheduler \
    --threshold_u 0.5 \
    --threshold_r 0.1 \
    --threshold_s 0.1 \
    --lambda_u 10 \
    --lambda_r 10 \
    --lambda_s 10 \
    --epsilon_loss 1e-8 \
    --seed 42 \
    --cuda_device 0 \
    --wandb_project pamap-multimodal-trus-moe

# python ../src/train_pamap.py \
#     --subject_id 2 \
#     --seq_len 100 \
#     --num_moe_layers 3 \
#     --cuda_device 0 \
#     --use_wandb \
#     --rus_max_lag 10 \
#     --rus_bins 8 \
#     --moe_num_experts 16 \
#     --moe_num_synergy_experts 2 \
#     --moe_k 2 \
#     --moe_expert_hidden_dim 128 \
#     --moe_router_gru_hidden_dim 64 \
#     --moe_router_token_processed_dim 64 \
#     --moe_router_attn_key_dim 32 \
#     --moe_router_attn_value_dim 32 \
#     --seed 42 \
#     --threshold_u 0.6 \
#     --threshold_r 0.6 \
#     --threshold_s 0.6 \
#     --lambda_u 0.1 \
#     --lambda_r 0.1 \
#     --lambda_s 0.1 \
#     --lambda_load 0.1 \
#     --plot_expert_activations \
#     --plot_num_samples 64