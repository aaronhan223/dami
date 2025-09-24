#! /bin/bash
# Best found: lambda_rus = 2.0, lambda_load = 0.08

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <checkpoint_path> <gpu>"
    echo "  checkpoint_path: Path to the model checkpoint"
    echo "  gpu: GPU device ID (e.g., 0, 1, 2)"
    echo ""
    echo "Example: $0 ../results/wesad/checkpoints/wesad_lambdarus2.0_lambdaload0.12/best_multimodal_model_wesad.pth 0"
    exit 1
fi

# Get arguments
CHECKPOINT_PATH=$1
GPU=$2

python ../src/test_wesad_multimodal.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --processed_dataset_dir /home/hhchung/data/WESAD_processed/ \
    --rus_data_path ../results/wesad/rus_multimodal_all_lag10.npy \
    --gpu $GPU \
    --eval_train \
    --eval_val \
    --plot_expert_activations \
    --plot_num_samples 1024 \
    --save_metrics