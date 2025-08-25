#!/bin/bash

cd /cis/home/xhan56/code/dami/src

# Set CUDA device for GPU 3
export CUDA_VISIBLE_DEVICES=3

python pamap_temporal_distribution_tracking_rus.py \
--subject_id 3 \
--batch_comparison_file ../results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy \
--plot_results \
--max_lag 10 \
--update_frequency 1 \
--n_components 8 \
--mc_samples 5000 \
--use_wandb \
--experiment_name "rus_estimate_gpu3_components8" \
--cuda_device 3