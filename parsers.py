import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # model_info
    parser.add_argument('--method', default='SaCCM')    # We privide the implements of SaCCM, SaTESAC, CCM and TESAC.
    parser.add_argument('--adversarial_loss_coef', default=0.01, type=float)
    parser.add_argument('--batch_size',default=16,type=int)
    parser.add_argument('--contrast_batch_size',default=16,type=int)
    parser.add_argument('--contrast_buffer_type',default='fine_grained',type=str)
    parser.add_argument('--contrast_training_interval',default=1,type=int)
    # train_info
    parser.add_argument('--time_step', default=500_000, type=int)   # timestep for each environment
    parser.add_argument('--causal_dim', default=6, type=int)
    parser.add_argument('--causal_hidden_dim', default=128, type=int)
    parser.add_argument('--buffer_size', default=100, type=int)
    parser.add_argument('--use_weighted_info_nce',action="store_true", default=False)
    parser.add_argument('--use_reward_norm',action="store_true", default=False)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--train_freq', default=128, type=int)
    parser.add_argument('--gradient_steps', default=16, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--neg_radio', default=0.5, type=float)
    # env
    parser.add_argument('--env_name', default='PandaPush-v3', type=str)
    parser.add_argument('--env_hook', default='PandaHook', type=str)
    parser.add_argument('--train_envs', default="[(0, 1), (0, 30),(1, 1),(1, 5),(1, 30),(10, 1),(10, 5),(30, 1)]", type=str, help="[(friction, mass),...]") # 
    parser.add_argument('--test_envs', default="[]", type=str, help="[(friction, mass),...]") # 

    # train
    parser.add_argument('--use_continue_train',action="store_true", default=False)  # set "use_continue_train=True" to load a exist model and continue training
    # parser.add_argument('--use_continue_train', default=True, type=bool)
    parser.add_argument('--save_video',action="store_true", default=False)
    # parser.add_argument('--save_video',default=True, type=bool)
    parser.add_argument('--seed',default=100, type=int)
    # parser.add_argument('--config_path',default = '/home/lixin/work/rl_learning/rl_learning/output/2024-01-17-03:21:53-202320-env1-OUR-last', type=str)
    parser.add_argument('--config_path',default = '', type=str)
    parser.add_argument('--test_eps_num_per_env',default=100, type=int)
    parser.add_argument('--use_wandb',action="store_true", default=False)
    parser.add_argument('--wandb_project_name',default = 'skill-aware-rl', type=str)
    parser.add_argument('--wandb_team_name',default = 'skill-aware-rl', type=str)

    args = parser.parse_args()
    if args.method == 'SAC':
        args.causal_dim = -1
        args.causal_hidden_dim = -1
    return args