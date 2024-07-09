#!/bin/bash

#SBATCH --job-name=hyper-t      #作业名称
#SBATCH --ntasks=8
#SBATCH --time=100:00:00     #申请运行时间
#SBATCH --output=./outputs/%j.out                #作业标准输出
#SBATCH --error=./outputs/%j.err                   #作业标准报错信息
#SBATCH --gres=gpu:1                   #申请1张GPU卡

# source ~/.bashrc     #激活conda环境
# conda activate dmcontrol
# python tools/retest_models.py --date_start 2024-04-03-08:43:58-532505  --date_end 2024-04-03-08:43:58-532505

# ############################################################################################################

## TESAC 0
conda run -n dmcontrol python main.py \
    --env_name AntEnv \
    --env_hook DominoHook \
    --method TESAC \
    --seed 150 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --use_wandb \
    --test_envs "[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]" &

conda run -n dmcontrol python main.py \
    --env_name HalfCheetahEnv \
    --env_hook DominoHook \
    --method TESAC \
    --seed 150 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --use_wandb \
    --test_envs "[([0.40, 0.50],[0.40, 0.50]),([0.40, 0.50],[1.50, 1.60]),([1.50, 1.60],[0.40, 0.50]),([1.50, 1.60],[1.50, 1.60])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]" &

conda run -n dmcontrol python main.py \
    --env_name CrippleAntEnv \
    --env_hook DominoHook \
    --method TESAC \
    --seed 150 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([3], [0],[0.4, 0.5]),([3], [0],[1.5, 1.6])]" \
    --test_eps_num_per_env 5 \
    --use_wandb \
    --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [0.75,0.85]),([2], [0], [0.75,0.85]),([2], [0], [1.0,1.15,1.25])]" &

conda run -n dmcontrol python main.py \
    --env_name CrippleHalfCheetahEnv \
    --env_hook DominoHook \
    --method TESAC \
    --seed 150 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --use_wandb \
    --test_envs "[([4, 5], [0], [0.4, 0.5]),([4, 5], [0],[1.5, 1.6])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [1.0,1.15,1.25]),([2, 3], [0], [0.75,0.85]),([2, 3], [0], [1.0,1.15,1.25])]" &

conda run -n dmcontrol python main.py \
    --env_name SlimHumanoidEnv \
    --env_hook DominoHook \
    --method TESAC \
    --use_wandb \
    --seed 150 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0.60, 0.70], [0.60, 0.70]),([0.60, 0.70], [1.50, 1.60]),([1.50, 1.60], [0.60, 0.70]),([1.50, 1.60], [1.50, 1.60])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0.8, 0.9], [0.8, 0.9]),([0.8, 0.9], [1.0, 1.15, 1.25]),([1.0, 1.15, 1.25], [0.8, 0.9]),([1.0, 1.15, 1.25], [1.0, 1.15, 1.25])]" &

conda run -n dmcontrol python main.py \
    --env_name HopperEnv \
    --env_hook DominoHook \
    --method TESAC \
    --use_wandb \
    --seed 150 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0.25, 0.375], [0.25, 0.375]),([0.25, 0.375], [1.75, 2.0]),([1.75, 2.0], [0.25, 0.375]),([1.75, 2.0], [1.75, 2.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0.5, 0.75, 1.0], [0.5, 0.75, 1.0]),([0.5, 0.75, 1.0], [1.25, 1.5]),([1.25, 1.5], [0.5, 0.75, 1.0]),([1.25, 1.5], [1.25, 1.5])]" 

# ### CCM 0
# conda run -n dmcontrol python main.py \
#     --env_name AntEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name HalfCheetahEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([0.40, 0.50],[0.40, 0.50]),([0.40, 0.50],[1.50, 1.60]),([1.50, 1.60],[0.40, 0.50]),([1.50, 1.60],[1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([3], [0],[0.4, 0.5]),([3], [0],[1.5, 1.6])]" \
#     --test_eps_num_per_env 5 \
#     --use_wandb \
#     --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [0.75,0.85]),([2], [0], [0.75,0.85]),([2], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([4, 5], [0], [0.4, 0.5]),([4, 5], [0],[1.5, 1.6])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [1.0,1.15,1.25]),([2, 3], [0], [0.75,0.85]),([2, 3], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name SlimHumanoidEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --use_wandb \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.60, 0.70], [0.60, 0.70]),([0.60, 0.70], [1.50, 1.60]),([1.50, 1.60], [0.60, 0.70]),([1.50, 1.60], [1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.8, 0.9], [0.8, 0.9]),([0.8, 0.9], [1.0, 1.15, 1.25]),([1.0, 1.15, 1.25], [0.8, 0.9]),([1.0, 1.15, 1.25], [1.0, 1.15, 1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --use_wandb \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.25, 0.375], [0.25, 0.375]),([0.25, 0.375], [1.75, 2.0]),([1.75, 2.0], [0.25, 0.375]),([1.75, 2.0], [1.75, 2.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.5, 0.75, 1.0], [0.5, 0.75, 1.0]),([0.5, 0.75, 1.0], [1.25, 1.5]),([1.25, 1.5], [0.5, 0.75, 1.0]),([1.25, 1.5], [1.25, 1.5])]" 


# ### SaCCM 0
# conda run -n dmcontrol python main.py \
#     --env_name AntEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]"  &

# conda run -n dmcontrol python main.py \
#     --env_name HalfCheetahEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([0.40, 0.50],[0.40, 0.50]),([0.40, 0.50],[1.50, 1.60]),([1.50, 1.60],[0.40, 0.50]),([1.50, 1.60],[1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]"  &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([3], [0],[0.4, 0.5]),([3], [0],[1.5, 1.6])]" \
#     --test_eps_num_per_env 5 \
#     --use_wandb \
#     --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [0.75,0.85]),([2], [0], [0.75,0.85]),([2], [0], [1.0,1.15,1.25])]"  &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([4, 5], [0], [0.4, 0.5]),([4, 5], [0],[1.5, 1.6])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [1.0,1.15,1.25]),([2, 3], [0], [0.75,0.85]),([2, 3], [0], [1.0,1.15,1.25])]"  &

# conda run -n dmcontrol python main.py \
#     --env_name SlimHumanoidEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --use_wandb \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.60, 0.70], [0.60, 0.70]),([0.60, 0.70], [1.50, 1.60]),([1.50, 1.60], [0.60, 0.70]),([1.50, 1.60], [1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.8, 0.9], [0.8, 0.9]),([0.8, 0.9], [1.0, 1.15, 1.25]),([1.0, 1.15, 1.25], [0.8, 0.9]),([1.0, 1.15, 1.25], [1.0, 1.15, 1.25])]"  &

# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --use_wandb \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.25, 0.375], [0.25, 0.375]),([0.25, 0.375], [1.75, 2.0]),([1.75, 2.0], [0.25, 0.375]),([1.75, 2.0], [1.75, 2.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.5, 0.75, 1.0], [0.5, 0.75, 1.0]),([0.5, 0.75, 1.0], [1.25, 1.5]),([1.25, 1.5], [0.5, 0.75, 1.0]),([1.25, 1.5], [1.25, 1.5])]"  

# # ## SaSAC 0
# conda run -n dmcontrol python main.py \
#     --env_name AntEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name HalfCheetahEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([0.40, 0.50],[0.40, 0.50]),([0.40, 0.50],[1.50, 1.60]),([1.50, 1.60],[0.40, 0.50]),([1.50, 1.60],[1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([3], [0],[0.4, 0.5]),([3], [0],[1.5, 1.6])]" \
#     --test_eps_num_per_env 5 \
#     --use_wandb \
#     --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [0.75,0.85]),([2], [0], [0.75,0.85]),([2], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --test_envs "[([4, 5], [0], [0.4, 0.5]),([4, 5], [0],[1.5, 1.6])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0, 1], [0], [0.75,0.85]),([0, 1], [0], [1.0,1.15,1.25]),([2, 3], [0], [0.75,0.85]),([2, 3], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name SlimHumanoidEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --use_wandb \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.60, 0.70], [0.60, 0.70]),([0.60, 0.70], [1.50, 1.60]),([1.50, 1.60], [0.60, 0.70]),([1.50, 1.60], [1.50, 1.60])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.8, 0.9], [0.8, 0.9]),([0.8, 0.9], [1.0, 1.15, 1.25]),([1.0, 1.15, 1.25], [0.8, 0.9]),([1.0, 1.15, 1.25], [1.0, 1.15, 1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.25, 0.375], [0.25, 0.375]),([0.25, 0.375], [1.75, 2.0]),([1.75, 2.0], [0.25, 0.375]),([1.75, 2.0], [1.75, 2.0])]" \
#     --test_eps_num_per_env 5 \
#     --use_wandb \
#     --train_envs "[([0.5, 0.75, 1.0], [0.5, 0.75, 1.0]),([0.5, 0.75, 1.0], [1.25, 1.5]),([1.25, 1.5], [0.5, 0.75, 1.0]),([1.25, 1.5], [1.25, 1.5])]" 


