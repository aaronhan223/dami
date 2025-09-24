#! /bin/bash

# Best found: lambda_rus = 2.0, lambda_load = 0.08

# Check if correct number of arguments provided
if [ $# -ne 6 ]; then
    echo "Usage: $0 <lambda_u> <lambda_r> <lambda_s> <lambda_load> <gpu> <seed>"
    echo "  lambda_u: Weight for uniqueness loss (e.g., 0, 0.5, 1)"
    echo "  lambda_r: Weight for redundancy loss (e.g., 0, 0.5, 1)"
    echo "  lambda_s: Weight for synergy loss (e.g., 0, 0.5, 1)"
    echo "  lambda_load: Weight for load-balancing loss (e.g., 0.02, 0.08)"
    echo "  gpu: GPU device ID (e.g., 0, 1, 2)"
    echo "  seed: Random seed (e.g., 42)"
    echo ""
    echo "Example: $0 0.5 1 0.1 0.08 1 42"
    exit 1
fi

# Get arguments
LAMBDA_U=$1
LAMBDA_R=$2
LAMBDA_S=$3
LAMBDA_LOAD=$4
GPU=$5
SEED=$6


# Validate lambda_u/r/s and lambda_load arguments
number_regex='^[0-9]*\.?[0-9]+$'
if [[ ! "$LAMBDA_U" =~ $number_regex ]]; then
    echo "Error: lambda_u must be a number (got: $LAMBDA_U)"
    exit 1
fi
if [[ ! "$LAMBDA_R" =~ $number_regex ]]; then
    echo "Error: lambda_r must be a number (got: $LAMBDA_R)"
    exit 1
fi
if [[ ! "$LAMBDA_S" =~ $number_regex ]]; then
    echo "Error: lambda_s must be a number (got: $LAMBDA_S)"
    exit 1
fi
if [[ ! "$LAMBDA_LOAD" =~ $number_regex ]]; then
    echo "Error: lambda_load must be a number (got: $LAMBDA_LOAD)"
    exit 1
fi

# Validate GPU argument
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "Error: gpu must be a non-negative integer (got: $GPU)"
    exit 1
fi

# Validate seed argument
if [[ ! "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "Error: seed must be an integer (got: $SEED)"
    exit 1
fi

RUN_NAME="wesad_u${LAMBDA_U}_r${LAMBDA_R}_s${LAMBDA_S}_load${LAMBDA_LOAD}_seed${SEED}"

echo "Starting training with:"
echo "  lambda_u = $LAMBDA_U"
echo "  lambda_r = $LAMBDA_R"
echo "  lambda_s = $LAMBDA_S"
echo "  lambda_load = $LAMBDA_LOAD"
echo "  gpu = $GPU"
echo "  seed = $SEED"
echo "  wandb_run_name = $RUN_NAME"
echo ""

python ../src/train_wesad_multimodal.py \
    --processed_dataset_dir /home/hhchung/data/WESAD_processed/ \
    --rus_data_path ../results/wesad/rus_multimodal_all_lag10.npy \
    --gpu $GPU \
    --seed $SEED \
    --use_wandb \
    --wandb_project wesad-multimodal-trus-moe \
    --lambda_u $LAMBDA_U \
    --lambda_r $LAMBDA_R \
    --lambda_s $LAMBDA_S \
    --lambda_load $LAMBDA_LOAD \
    --run_name $RUN_NAME \
    --epochs 30