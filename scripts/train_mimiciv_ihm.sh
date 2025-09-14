#! /bin/bash

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <lambda_rus> <lambda_load> <gpu>"
    echo "  lambda_rus: Value for lambda_u, lambda_r, and lambda_s (e.g., 0, 0.5, 1)"
    echo "  lambda_load: Value for lambda_load (e.g., 0.02, 0.05)"
    echo "  gpu: GPU device ID (e.g., 0, 1, 2)"
    echo ""
    echo "Example: $0 0.5 0.02 1"
    exit 1
fi

# Get arguments
LAMBDA_RUS=$1
LAMBDA_LOAD=$2
GPU=$3

# Validate lambda_rus argument
if [[ ! "$LAMBDA_RUS" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Error: lambda_rus must be a number (got: $LAMBDA_RUS)"
    exit 1
fi

# Validate lambda_load argument
if [[ ! "$LAMBDA_LOAD" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Error: lambda_load must be a number (got: $LAMBDA_LOAD)"
    exit 1
fi

# Validate GPU argument
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "Error: gpu must be a non-negative integer (got: $GPU)"
    exit 1
fi

# Create run name with hyperparameters
RUN_NAME="mimiciv_ihm_lambdarus${LAMBDA_RUS}_lambdaload${LAMBDA_LOAD}_gpu${GPU}"

echo "Starting training with:"
echo "  lambda_u = lambda_r = lambda_s = $LAMBDA_RUS"
echo "  lambda_load = $LAMBDA_LOAD"
echo "  gpu = $GPU"
echo "  wandb_run_name = $RUN_NAME"
echo ""

python ../src/train_mimiciv_multimodal.py \
    --train_data_path ../../mimic-iv-preprocess/data/ihm/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl \
    --val_data_path ../../mimic-iv-preprocess/data/ihm/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl \
    --rus_data_path ../results/mimiciv/ihm/rus_multimodal_all_meanpool.npy \
    --task ihm \
    --truncate_from_end \
    --gpu $GPU \
    --use_wandb \
    --wandb_project mimiciv-multimodal-trus-moe \
    --plot_expert_activations \
    --lambda_u $LAMBDA_RUS \
    --lambda_r $LAMBDA_RUS \
    --lambda_s $LAMBDA_RUS \
    --lambda_load $LAMBDA_LOAD \
    --wandb_run_name $RUN_NAME \
    --lr 1e-3 \
    --epochs 20