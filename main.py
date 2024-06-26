#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Author  :   Lixin & Xuehui
@Version :   1.0
@Desc    :   Training & Zero-shot generalisation (Test)
'''
import os
from logger import Manager
from tools.utils import set_random_seed

### environment
from envs import PandaHook,BaseHook,DominoHook,ENV_MAX_STEP
from stable_baselines3.common.vec_env import DummyVecEnv

### methods
from algorithm import get_model
from test_model import test_model
from parsers import get_args

if __name__ == '__main__':
    # prepare log or config
    project_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_dir, 'output')
    args = get_args()
    manager = Manager(args, log_dir)
    if os.path.exists(args.config_path):
        manager.load_config(args.config_path)
    if args.config_path != '' and os.path.exists(os.path.join(project_dir, args.config_path)):
        manager.load_config(os.path.join(project_dir, args.config_path))
    manager.setup()
    max_step = ENV_MAX_STEP[manager.model_parameters['env_name']]
    hook:BaseHook = eval(manager.model_parameters['env_hook'])()
    try:
        if not os.path.exists(manager.save_model_path) or \
            manager.model_parameters['use_continue_train']:
            # prepare environment
            envs_info = eval(manager.model_parameters['train_envs'])
            envs = []
            for _env_info in envs_info:
                envs.append(hook.make_env(manager,_env_info))
            train_env = DummyVecEnv(envs)
            # prepare model
            model = get_model(manager,train_env,hook.causal_keys,max_step)
            model.set_logger(manager.setup_logger())
            total_timesteps = int(manager.model_parameters['time_step'] * len(envs_info))
            reset_num_timesteps = True
            for i in range(total_timesteps // 1_00_000):
                model.learn(total_timesteps=1_00_000,
                            progress_bar=True,
                            reset_num_timesteps=reset_num_timesteps)
                reset_num_timesteps = False
                if manager.model_parameters['use_continue_train'] and (i % 8 == 0 or i == total_timesteps // 1_00_000-1):
                    model.save(os.path.join(manager.sub_save_path, f'model_continue_{i}.zip'))
                elif i % 8 == 0 or i == total_timesteps // 1_00_000-1:
                    model.save(os.path.join(manager.sub_save_path, f'model_{i}.zip'))
                
            test_model(model, manager,hook, (i+1) * 1_00_000)
            train_env.close()
        else:
            model = get_model(manager,causal_keys=hook.causal_keys,max_step_num=max_step)
            model.set_logger(manager.setup_logger())
            test_model(model, manager, hook)
        
    except Exception as e:
        print(e)
        raise e
    finally:
        manager.close()
