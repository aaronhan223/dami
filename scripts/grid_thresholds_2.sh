#!/bin/bash

# Grid search for threshold parameters: threshold_u, threshold_r, threshold_s
# Values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

cd /cis/home/xhan56/code/dami/src


for threshold_u in 0.1 0.2 0.3 0.4 0.5 0.6; do
    for threshold_r in 0.1 0.2 0.3 0.4 0.5 0.6; do
        for threshold_s in 0.1 0.2 0.3 0.4 0.5 0.6; do
            echo "Running with threshold_u=$threshold_u, threshold_r=$threshold_r, threshold_s=$threshold_s"
            python train_pamap_multimodal.py \
                --subject_id 2 \
                --use_wandb \
                --cuda_device 1 \
                --wandb_project pamap-grid-search \
                --wandb_run_name "u=$threshold_u,r=$threshold_r,s=$threshold_s,id=2" \
                --threshold_u $threshold_u \
                --threshold_r $threshold_r \
                --threshold_s $threshold_s
        done
    done
done