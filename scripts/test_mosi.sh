#! /bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <checkpoint_path> <gpu>"
    echo "  checkpoint_path: Path to the model checkpoint"
    echo "  gpu: GPU device ID (e.g., 0, 1, 2)"
    echo ""
    echo "Example: $0 ../results/affect/mosi/checkpoints/mosi_lambdarus2.0_lambdaload0.05/best_multimodal_model_mosi.pth 0"
    exit 1
fi

# Get arguments
CHECKPOINT_PATH=$1
GPU=$2

python ../src/test_mosi_multimodal.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --dataset_path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/hhchung/mosi/mosi_raw.pkl \
    --rus_data_path ../results/affect/mosi/rus_multimodal_all_seq50_lags10_meanpool.npy \
    --dataset mosi \
    --gpu $GPU \
    --eval_train \
    --eval_val