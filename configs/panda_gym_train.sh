# panda
CUDA_VISIBLE_DEVICES=0 python main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaCCM \
    --buffer_size 1000 \
    --train_freq 128 \
    --gradient_steps 16 \
    --learning_rate 1e-3 \
    --batch_size 256 \
    --contrast_batch_size 256 \
    --encoder_tau 0.05 \
    --seed 100 \
    --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
    --test_eps_num_per_env 50 \
    --use_wandb \
    --time_step 1_000_000 \
    --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5)]"