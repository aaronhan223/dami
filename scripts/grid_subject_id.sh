#!/bin/bash

# Grid search for subject_id parameter (baseline)
# Values: [1, 2, 3]

cd /cis/home/xhan56/code/dami/src

# for subject_id in 1 2 3; do
#     echo "Running with subject_id=$subject_id"
#     python train_pamap_multimodal_baseline.py \
#         --subject_id $subject_id \
#         --use_wandb \
#         --cuda_device 2 \
#         --wandb_project pamap-grid-search \
#         --wandb_run_name "baseline_id=$subject_id,lr=1e-3"
# done


# for subject_id in 1 2 3; do
#     echo "Running with subject_id=$subject_id"
#     python train_pamap_multimodal.py \
#         --subject_id $subject_id \
#         --use_wandb \
#         --cuda_device 2 \
#         --wandb_project pamap-grid-search \
#         --wandb_run_name "multimodal_id=$subject_id"
# done

python train_pamap_multimodal.py \
    --subject_id 3 \
    --cuda_device 2 \
    --threshold_u 0.4 \
    --threshold_r 0.3 \
    --threshold_s 0.5 \
    --plot_expert_activations