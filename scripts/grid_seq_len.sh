#!/bin/bash

# Grid search for seq_len parameter
# Values: [10, 20, 50, 70, 100]

cd /cis/home/xhan56/code/dami/src

for seq_len in 10 20 50 70 100; do
    echo "Running with seq_len=$seq_len"
    python train_pamap_multimodal.py \
        --subject_id 1 \
        --use_wandb \
        --cuda_device 1 \
        --wandb_project pamap-grid-search \
        --wandb_run_name "seq_len=$seq_len,id=1" \
        --seq_len $seq_len
done

for seq_len in 10 20 50 70 100; do
    echo "Running with seq_len=$seq_len"
    python train_pamap_multimodal.py \
        --subject_id 2 \
        --use_wandb \
        --cuda_device 1 \
        --wandb_project pamap-grid-search \
        --wandb_run_name "seq_len=$seq_len,id=2" \
        --seq_len $seq_len
done

for seq_len in 10 20 50 70 100; do
    echo "Running with seq_len=$seq_len"
    python train_pamap_multimodal.py \
        --subject_id 3 \
        --use_wandb \
        --cuda_device 1 \
        --wandb_project pamap-grid-search \
        --wandb_run_name "seq_len=$seq_len,id=3" \
        --seq_len $seq_len
done