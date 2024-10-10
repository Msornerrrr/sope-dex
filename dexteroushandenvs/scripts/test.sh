seed=4400
runs=10000

python train.py \
  --task=AllegroVertical \
  --algo=ppo \
  --num_envs=1 \
  --seed 999  \
  --play \
  --model_dir=logs/allegro_vertical/ppo/ppo_seed$seed/model_$runs.pt \
  # --pipeline=cpu
