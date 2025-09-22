#! /bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <checkpoint_path> <gpu>"
    echo "  checkpoint_path: Path to the model checkpoint"
    echo "  gpu: GPU device ID (e.g., 0, 1, 2)"
    echo ""
    echo "Example: $0 ../results/mimiciv/los/checkpoints/mimiciv_los_lambdarus1_lambdaload0.02/best_multimodal_model_mimiciv.pth 0"
    exit 1
fi

# Get arguments
CHECKPOINT_PATH=$1
GPU=$2

python ../src/test_mimiciv_multimodal.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --test_data_path /home/hhchung/mimic-iv-preprocess/data/los/test_los-cxr-notes-missingInd-standardized_stays.pkl \
    --rus_data_path ../results/mimiciv/los/rus_multimodal_all_seq48_lags8_meanpool.npy \
    --gpu $GPU \
    --eval_train \
    --eval_val