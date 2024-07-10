import os
import yaml
import sys
import itertools
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from stable_baselines3.common.vec_env import DummyVecEnv

from logger import Manager

def next_observation(model, prev_observations,actions,observations, dones):
    """
    Return: observation
        observations['causal']: context embedding
        observations['hidden_h']: hidden state
        observations['hidden_c']: cell state

    prev_observations: Previous observations
    actions: Actions
    observations: Observations
    dones: Dones
    """
    if 'hidden_h' in observations:
        # for rnn 
        # reset next obs hidden_h and hidden_c
        _observations = OrderedDict()
        actions = model.policy.scale_action(actions)
        _observations['action'] = (actions * (1-np.stack((dones,), axis = -1))).astype(np.float32)
        for key in observations:
            _observations[key] = observations[key].astype(np.float32)
        _observations['hidden_h'] = prev_observations['hidden_h']
        _observations['hidden_c'] = prev_observations['hidden_c']
        causal, hidden_h, hidden_c = model.policy.rnn_encoder_predict(_observations)
        observations['causal'] = causal.astype(np.float32)
        observations['hidden_h'] = hidden_h * (1-np.stack((dones,), axis = -1)).astype(np.float32)
        observations['hidden_c'] = hidden_c * (1-np.stack((dones,), axis = -1)).astype(np.float32)
    
    return observations

def test_model(model, manager:Manager, hook, time_steps=-1):

    # #############hook init#############
    hook.start_test(manager.model_parameters['train_envs'],test_envs = manager.model_parameters['test_envs'])
    # #############hook init#############
    tsne_x,tsne_y,tsne_c,tsne_alpha = [],[],[],[]
    for env_i, _env_info in tqdm(enumerate(hook.test_envs)):
        # test env
        env = hook.make_env(manager, _env_info)
        test_env = DummyVecEnv([env])

        # ###########hook env start###########
        hook.start_env(_env_info)
        # ###########hook env start###########

        if manager.model_parameters['save_video']:
            manager.enable_video()
        else:
            manager.disable_video()
        
        while len(hook.test_infos[hook.encoder_env_info(_env_info)]['eps_states']) < manager.model_parameters['test_eps_num_per_env']:
            observations = test_env.reset()
            states = None
            # episode_starts = np.ones((test_env.num_envs,), dtype=bool)
            _eps_states = []
            manager.reset_video()
            for eps_i in range(hook.max_step_num):
                manager.record_video(test_env)
                actions, states = model.predict(
                    observations,
                    state=states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                prev_observations = copy.deepcopy(observations)
                observations, rewards, dones, infos = test_env.step(actions)
                observations = next_observation(model,prev_observations,actions,observations, dones)
                
                # if ((eps_i+1) % 2 ==0 or dones) and 'hidden_h' in observations:
                if (dones) and 'hidden_h' in observations:
                    tsne_x.append(observations['causal'])
                    tsne_y.append(env_i)
                    tsne_alpha.append(1.0)
                    # tsne_alpha.append(min(eps_i/hook.max_step_num * 5, 1.0))
                    class_name = hook.encoder_env_info(_env_info)
                    if class_name not in tsne_c:
                        tsne_c.append(class_name)

                if not dones:
                    _eps_states.append(hook.get_state(test_env, infos))
                else:
                    if infos[0]['is_success']:
                        _eps_states.append('success')
                    else:
                        _eps_states.append('fail')
                    break
        
            # if _eps_states[-1] == 'success':
            manager.save_video(f'{str(_env_info)}-{len(hook.test_infos[hook.encoder_env_info(_env_info)]["eps_states"])}.mp4')
            # manager.disable_video()

            # ###########hook eps end###########
            hook.end_eps(_env_info, _eps_states)
            # ###########hook eps end###########

            # if cur_tsne == per_tsne: break

        # ###########hook env end###########
        hook.end_env(_env_info, model.logger)
        # ###########hook env end###########

        sys.stdout.flush()
        test_env.close()

    # if len(tsne_x) > 0: # 绘制tsne
    #     manager.plot_scatter(np.concatenate(tsne_x,axis=0),np.array(tsne_y),tsne_c,np.array(tsne_alpha))
    # ###########hook end###########
    hook.end_hook(manager, time_steps)
    # ###########hook end###########
