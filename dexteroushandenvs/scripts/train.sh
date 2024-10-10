#!/bin/bash

# Export CUDA device
# export CUDA_VISIBLE_DEVICES=1

# Define seeds
seeds=(4400)

# Loop through each seed and run the training script
for seed in "${seeds[@]}"
do
    echo "Training with seed $seed..."
    python train.py \
        --task=AllegroVertical \
        --algo=ppo \
        --num_envs=8192 \
        --seed $seed \
        --headless
done

# Wait for all processes to complete
wait

echo "Training completed for all seeds."
