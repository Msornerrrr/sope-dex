#!/bin/bash

# Export CUDA device
# export CUDA_VISIBLE_DEVICES=1

seed=4400
runs=10000
new_seed=4401

python train.py \
  --task=AllegroVertical \
  --algo=ppo \
  --num_envs=8192 \
  --seed $new_seed  \
  --model_dir=logs/allegro_vertical/ppo/ppo_seed$seed/model_$runs.pt \
  --headless

# --model_dir=logs/allegro_vertical/ppo/ppo_seed$seed/model_$runs.pt \
# --model_dir=models/week-06-04/cornerstate_phase3_20000.pt \
# --model_dir=models/tmp/887_model_20000.pt \
