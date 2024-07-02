# domino [ant env]
CUDA_VISIBLE_DEVICES=0 python main.py \
    --env_name AntEnv \
    --env_hook DominoHook \
    --method SaCCM \
    --adversarial_loss_coef 0.1 \
    --use_weighted_info_nce \
    --seed 421 \
    --use_wandb \
    --test_envs "[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]"

