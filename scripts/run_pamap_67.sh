#!/bin/bash

python ../src/pamap_rus_multimodal.py \
    --method batch \
    --subject_id 6 \
    --max_lag 10 \
    --dominance_threshold 0.4 \
    --dominance_percentage 0.9 \
    --gpu 1 \
    --hidden_dim 64 \
    --layers 3 \
    --lr 0.001 \
    --discrim_epochs 30 \
    --ce_epochs 15 \
    --activation relu \
    --embed_dim 20 \
    --batch_size 512 \
    --n_batches 3 \
    --seed 42

python ../src/pamap_rus_multimodal.py \
    --method batch \
    --subject_id 7 \
    --max_lag 10 \
    --dominance_threshold 0.4 \
    --dominance_percentage 0.9 \
    --gpu 1 \
    --hidden_dim 64 \
    --layers 3 \
    --lr 0.001 \
    --discrim_epochs 30 \
    --ce_epochs 15 \
    --activation relu \
    --embed_dim 20 \
    --batch_size 512 \
    --n_batches 3 \
    --seed 42
