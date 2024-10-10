seed=999

python train.py \
    --task=AllegroVertical \
    --algo=ppo \
    --num_env=64 \
    --seed $seed \
    # --pipeline=cpu 