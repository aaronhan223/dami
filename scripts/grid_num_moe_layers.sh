#!/bin/bash

# Grid search for num_moe_layers parameter
# Values: [1, 2, 3, 4]

cd /cis/home/xhan56/code/dami/src

for num_moe_layers in 1 2 3 4; do
    echo "Running with num_moe_layers=$num_moe_layers"
    python train_pamap_multimodal.py \
        --subject_id 1 \
        --use_wandb \
        --cuda_device 3 \
        --wandb_project pamap-grid-search \
        --wandb_run_name "num_moe_layers=$num_moe_layers,id=1" \
        --num_moe_layers $num_moe_layers
done

for num_moe_layers in 1 2 3 4; do
    echo "Running with num_moe_layers=$num_moe_layers"
    python train_pamap_multimodal.py \
        --subject_id 2 \
        --use_wandb \
        --cuda_device 3 \
        --wandb_project pamap-grid-search \
        --wandb_run_name "num_moe_layers=$num_moe_layers,id=2" \
        --num_moe_layers $num_moe_layers
done

for num_moe_layers in 1 2 3 4; do
    echo "Running with num_moe_layers=$num_moe_layers"
    python train_pamap_multimodal.py \
        --subject_id 3 \
        --use_wandb \
        --cuda_device 3 \
        --wandb_project pamap-grid-search \
        --wandb_run_name "num_moe_layers=$num_moe_layers,id=3" \
        --num_moe_layers $num_moe_layers
done