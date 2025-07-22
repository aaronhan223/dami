#!/bin/bash

# python ../src/train_pamap.py --subject_id 1 --seq_len 25 --num_moe_layers 1 --cuda_device 3 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 25 --num_moe_layers 2 --cuda_device 3 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 25 --num_moe_layers 3 --cuda_device 3 --use_wandb

# python ../src/train_pamap.py --subject_id 7 --seq_len 100 --num_moe_layers 3 --cuda_device 3 --use_wandb --rus_dominance_threshold 0.4

# python ../src/train_pamap.py --subject_id 8 --seq_len 100 --num_moe_layers 3 --cuda_device 3 --use_wandb --rus_dominance_threshold 0.4

# python ../src/train_pamap_baseline.py \
#     --subject_id 2 \
#     --seq_len 100 \
#     --num_moe_layers 3 \
#     --cuda_device 3 \
#     --use_wandb \
#     --moe_num_experts 16 \
#     --moe_k 2 \
#     --moe_expert_hidden_dim 128 \
#     --seed 42 \
#     --lambda_load 0.1

python ../src/pamap_rus_multimodal.py \
    --method batch \
    --subject_id 4 \
    --max_lag 10 \
    --dominance_threshold 0.4 \
    --dominance_percentage 0.9 \
    --gpu 3 \
    --hidden_dim 64 \
    --layers 3 \
    --lr 0.001 \
    --discrim_epochs 30 \
    --ce_epochs 15 \
    --activation relu \
    --embed_dim 20 \
    --batch_size 512 \
    --n_batches 3 \
    --seed 42

python ../src/pamap_rus_multimodal.py \
    --method batch \
    --subject_id 5 \
    --max_lag 10 \
    --dominance_threshold 0.4 \
    --dominance_percentage 0.9 \
    --gpu 3 \
    --hidden_dim 64 \
    --layers 3 \
    --lr 0.001 \
    --discrim_epochs 30 \
    --ce_epochs 15 \
    --activation relu \
    --embed_dim 20 \
    --batch_size 512 \
    --n_batches 3 \
    --seed 42
