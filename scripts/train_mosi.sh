#! /bin/bash

# Best found: lambda_rus = 2.0, lambda_load = 0.02
# Best found: lambda_rus = 2.0, lambda_load = 0.05
# Best found: lambda_rus = 2.0, lambda_load = 0.02, moe_num_synergy_experts = 4, num_encoder_layers = 2

# Check if correct number of arguments provided
if [ $# -ne 7 ]; then
    echo "Usage: $0 <lambda_u> <lambda_r> <lambda_s> <lambda_load> <moe_num_synergy_experts> <gpu> <seed>"
    echo "  lambda_u: Weight for uniqueness loss (e.g., 0, 0.5, 1)"
    echo "  lambda_r: Weight for redundancy loss (e.g., 0, 0.5, 1)"
    echo "  lambda_s: Weight for synergy loss (e.g., 0, 0.5, 1)"
    echo "  lambda_load: Weight for load-balancing loss (e.g., 0.02, 0.05)"
    echo "  moe_num_synergy_experts: Number of synergy experts (e.g., 1, 2, 4)"
    echo "  gpu: GPU device ID (e.g., 0, 1, 2)"
    echo "  seed: Random seed (e.g., 42)"
    echo ""
    echo "Example: $0 0.5 1 0.1 0.02 2 1 42"
    exit 1
fi

# Get arguments
LAMBDA_U=$1
LAMBDA_R=$2
LAMBDA_S=$3
LAMBDA_LOAD=$4
MOE_NUM_SYNERGY_EXPERTS=$5
GPU=$6
SEED=$7


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

# Validate moe_num_synergy_experts argument
if [[ ! "$MOE_NUM_SYNERGY_EXPERTS" =~ ^[0-9]+$ ]]; then
    echo "Error: moe_num_synergy_experts must be a positive integer (got: $MOE_NUM_SYNERGY_EXPERTS)"
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

RUN_NAME="mosi_u${LAMBDA_U}_r${LAMBDA_R}_s${LAMBDA_S}_load${LAMBDA_LOAD}_syn${MOE_NUM_SYNERGY_EXPERTS}_2enc_seed${SEED}"

echo "Starting training with:"
echo "  lambda_u = $LAMBDA_U"
echo "  lambda_r = $LAMBDA_R"
echo "  lambda_s = $LAMBDA_S"
echo "  lambda_load = $LAMBDA_LOAD"
echo "  moe_num_synergy_experts = $MOE_NUM_SYNERGY_EXPERTS"
echo "  gpu = $GPU"
echo "  seed = $SEED"
echo "  wandb_run_name = $RUN_NAME"
echo ""

python ../src/train_mosi_multimodal.py \
    --dataset_path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/hhchung/mosi/mosi_raw.pkl \
    --rus_data_path ../results/affect/mosi/rus_multimodal_all_seq50_lags10_meanpool.npy \
    --dataset mosi \
    --gpu $GPU \
    --seed $SEED \
    --use_wandb \
    --wandb_project mosi-multimodal-trus-moe \
    --lambda_u $LAMBDA_U \
    --lambda_r $LAMBDA_R \
    --lambda_s $LAMBDA_S \
    --lambda_load $LAMBDA_LOAD \
    --moe_num_synergy_experts $MOE_NUM_SYNERGY_EXPERTS \
    --run_name $RUN_NAME \
    --epochs 25