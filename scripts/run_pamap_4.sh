#!/bin/bash

# python ../src/train_pamap.py --subject_id 1 --seq_len 25 --num_moe_layers 1 --cuda_device 3 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 25 --num_moe_layers 2 --cuda_device 3 --use_wandb

# python ../src/train_pamap.py --subject_id 1 --seq_len 25 --num_moe_layers 3 --cuda_device 3 --use_wandb

python ../src/train_pamap.py --subject_id 7 --seq_len 100 --num_moe_layers 3 --cuda_device 3 --use_wandb --rus_dominance_threshold 0.4

python ../src/train_pamap.py --subject_id 8 --seq_len 100 --num_moe_layers 3 --cuda_device 3 --use_wandb --rus_dominance_threshold 0.4