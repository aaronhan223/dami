#!/bin/bash

cd /cis/home/xhan56/code/dami/src

# Set CUDA device for GPU 1
export CUDA_VISIBLE_DEVICES=1

# # Subject 1 experiment
python pamap_temporal_distribution_tracking_rus.py \
--subject_id 1 \
--batch_comparison_file ../results/pamap/pamap_subject1_multimodal_all_lag10_bins4.npy \
--plot_results \
--max_lag 10 \
--update_frequency 1 \
--n_components 8 \
--mc_samples 10000 \
--use_wandb \
--experiment_name "subject1_rus_estimate_gpu1_components8" \
--cuda_device 1

# # Subject 2 experiment
python pamap_temporal_distribution_tracking_rus.py \
--subject_id 2 \
--batch_comparison_file ../results/pamap/pamap_subject2_multimodal_all_lag10_bins4.npy \
--plot_results \
--max_lag 10 \
--update_frequency 1 \
--n_components 8 \
--mc_samples 10000 \
--use_wandb \
--experiment_name "subject2_rus_estimate_gpu1_components8" \
--cuda_device 1

# Subject 3 experiment
python pamap_temporal_distribution_tracking_rus.py \
--subject_id 3 \
--batch_comparison_file ../results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy \
--plot_results \
--max_lag 10 \
--update_frequency 1 \
--n_components 8 \
--mc_samples 10000 \
--use_wandb \
--experiment_name "subject3_rus_estimate_gpu1_components8" \
--cuda_device 1