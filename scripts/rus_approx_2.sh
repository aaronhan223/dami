#!/bin/bash

cd /cis/home/xhan56/code/dami/src

# Set CUDA device for GPU 2
export CUDA_VISIBLE_DEVICES=2

# Array of MI estimation methods to test
mi_methods=("knn" "neural" "hybrid")

# Array of n_components values to test (only for neural and hybrid methods)
n_components_values=(4 8 12 16 20 24 28 32)

# Create results directory for this experiment
mkdir -p ../results/pamap_method_comparison

echo "Starting method comparison for PAMAP experiment..."
echo "Testing MI methods: ${mi_methods[@]}"
echo "Testing n_components for neural/hybrid: ${n_components_values[@]}"

# Run experiments for each method
for method in "${mi_methods[@]}"
do
    echo "Testing MI method: $method"
    
    if [ "$method" == "knn" ]; then
        # KNN method doesn't use n_components, run once
        echo "Running KNN experiment (no n_components parameter)"
        
        python pamap_temporal_distribution_tracking_rus.py \
        --subject_id 3 \
        --batch_comparison_file ../results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy \
        --plot_results \
        --max_lag 10 \
        --update_frequency 1 \
        --mi_method $method \
        --n_neighbors 3 \
        --knn_threshold 10 \
        --use_wandb \
        --experiment_name "rus_estimate_gpu2_${method}" \
        --cuda_device 2 \
        --output_dir "../results/pamap_method_comparison"
        
        echo "Completed KNN experiment"
        
    else
        # Neural and hybrid methods: test different n_components values
        echo "Running $method experiments with different n_components values"
        
        for n_comp in "${n_components_values[@]}"
        do
            echo "Running $method experiment with n_components=$n_comp"
            
            python pamap_temporal_distribution_tracking_rus.py \
            --subject_id 3 \
            --batch_comparison_file ../results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy \
            --plot_results \
            --max_lag 10 \
            --update_frequency 1 \
            --mi_method $method \
            --n_components $n_comp \
            --mc_samples 5000 \
            --n_neighbors 3 \
            --knn_threshold 10 \
            --use_wandb \
            --experiment_name "rus_estimate_gpu2_${method}_components${n_comp}" \
            --cuda_device 2 \
            --output_dir "../results/pamap_method_comparison"
            
            echo "Completed $method experiment with n_components=$n_comp"
        done
    fi
done

echo "All method comparison experiments completed!"

# Generate comparison plots
echo "Generating method comparison plots..."

# Plot for neural method n_components comparison
echo "Generating neural method n_components comparison plot..."
python -c "
import sys
sys.path.append('/cis/home/xhan56/code/dami/src')
from plots.rus_estimation import plot_n_components_comparison
import os
os.makedirs('../results/pamap_method_comparison/neural_method', exist_ok=True)
plot_n_components_comparison('../results/pamap_method_comparison', '../results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy', '../results/pamap_method_comparison/neural_method', method_filter='neural')
"

# Plot for hybrid method n_components comparison
echo "Generating hybrid method n_components comparison plot..."
python -c "
import sys
sys.path.append('/cis/home/xhan56/code/dami/src')
from plots.rus_estimation import plot_n_components_comparison
import os
os.makedirs('../results/pamap_method_comparison/hybrid_method', exist_ok=True)
plot_n_components_comparison('../results/pamap_method_comparison', '../results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy', '../results/pamap_method_comparison/hybrid_method', method_filter='hybrid')
"

echo "All comparison plots generated!"