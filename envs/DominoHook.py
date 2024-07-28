import numpy as np
import itertools
import yaml

from .BaseHook import BaseHook
from .utils import make_env

# envs
from .ant_env import AntEnv # mass_scale_set damping_scale_set
from .half_cheetah_cripple_env import CrippleHalfCheetahEnv # cripple_set extreme_set mass_scale_set
from .ant_cripple_env import CrippleAntEnv # cripple_set extreme_set mass_scale_set
from .half_cheetah_env import HalfCheetahEnv # mass_scale_set damping_scale_set
from .humanoidstandup import HumanoidStandupEnv # mass_scale_set damping_scale_set
from .slim_humanoid_env import SlimHumanoidEnv # mass_scale_set damping_scale_set
from .hopper_env import HopperEnv # mass_scale_set damping_scale_set
from .hopper_cripple_env import CrippleHopperEnv # mass_scale_set damping_scale_set
from .walker2d import WalkerEnv # mass_scale_set damping_scale_set
from .walker2d_cripple_env import CrippleWalkerEnv # mass_scale_set damping_scale_set
from .walker_hopper2d import WalkerHopperEnv # mass_scale_set damping_scale_set
from .cartpole import RandomCartPole_Force_Length as Cartpoleenvs # force_set length_set
from .pendulum import RandomPendulumAll as Pendulumenvs # mass_set length_set

ENV_CLS = {
    'AntEnv':AntEnv,
    'CrippleHalfCheetahEnv':CrippleHalfCheetahEnv,
    'CrippleAntEnv':CrippleAntEnv,
    'HalfCheetahEnv':HalfCheetahEnv,
    'SlimHumanoidEnv':SlimHumanoidEnv,
    'HumanoidStandupEnv':HumanoidStandupEnv,
    'HopperEnv':HopperEnv,
    'CrippleHopperEnv':CrippleHopperEnv,
    'CrippleWalkerEnv':CrippleWalkerEnv,
    'WalkerEnv':WalkerEnv,
    'WalkerHopperEnv':WalkerHopperEnv,
    'Cartpoleenvs':Cartpoleenvs,
    'Pendulumenvs':Pendulumenvs
}
ENV_CAUSAL = {
    'AntEnv':['mass_scale_set','damping_scale_set'],
    'CrippleHalfCheetahEnv':['cripple_set','extreme_set','mass_scale_set'],
    'CrippleHopperEnv':['cripple_set','extreme_set','mass_scale_set'],
    'CrippleWalkerEnv':['cripple_set','extreme_set','mass_scale_set'],
    'WalkerHopperEnv':['cripple_set','extreme_set','mass_scale_set'],
    'CrippleAntEnv':['cripple_set','extreme_set','mass_scale_set'],
    'HalfCheetahEnv':['mass_scale_set','damping_scale_set'],
    'SlimHumanoidEnv':['mass_scale_set','damping_scale_set'],
    'HumanoidStandupEnv':['mass_scale_set','damping_scale_set'],
    'HopperEnv':['mass_scale_set','damping_scale_set'],
    'WalkerEnv':['mass_scale_set','damping_scale_set'],
    'Cartpoleenvs':['force_set','length_set'],
    'Pendulumenvs':['mass_set','length_set']
}

ENV_MAX_STEP = {
    'AntEnv': 1000,
    'CrippleHalfCheetahEnv':1000,
    'CrippleAntEnv':2000,
    'HalfCheetahEnv':1000,
    'SlimHumanoidEnv':2000,
    'HumanoidStandupEnv':2000,
    'HopperEnv':2000,
    'CrippleHopperEnv':1000,
    'CrippleWalkerEnv':2000,
    'WalkerHopperEnv':2000,
    'WalkerEnv':2000,
    'Cartpoleenvs':1000,
    'Pendulumenvs':1000
}

class DominoHook(BaseHook):
    def __init__(self) -> None:
        self.max_step_num = -1
        self.causal_keys = {'hidden_h','hidden_c'}

    def start_test(self, train_envs, test_envs=None):
        train_envs = eval(train_envs)
        self.train_envs = train_envs
        test_envs = eval(test_envs) if test_envs is not None else []
        self.test_envs = test_envs
        # self.test_envs = train_envs + test_envs
        
        self.reward_table = np.zeros((len(self.test_envs),1))
        self.test_infos = {} # save to test

    def start_env(self, env_info):
        test_info_key = self.encoder_env_info(env_info)
        self.test_infos[test_info_key] = {
            'reward':[],
            'traj_len': [],
            'eps_states': []
        }

    def end_eps(self,env_info, eps_states):
        test_info_key = self.encoder_env_info(env_info)
        self.test_infos[test_info_key]['reward'].append(eps_states[-2][0])
        self.test_infos[test_info_key]['traj_len'].append(eps_states[-2][1])
        self.test_infos[test_info_key]['eps_states'].append(1)

    def end_env(self, env_info, logger):
        test_info_key = self.encoder_env_info(env_info)
        logger.info('#' * 10 + f"{env_info}" + '#'*10)
        logger.info(f'reward {np.mean(self.test_infos[test_info_key]["reward"])}')
        logger.info(f'traj_len {np.mean(self.test_infos[test_info_key]["traj_len"])}')
        logger.info('#' * 19)
        
    def end_hook(self, manager, time_steps):
        pass

    def get_state(self,envs,env_info=None):
        return env_info[0]['episode']['l'],env_info[0]['episode']['r']

    def make_env(self, manager, env_info):
        env_name = manager.model_parameters['env_name']
        self.max_step_num = ENV_MAX_STEP[env_name]
        keys = ENV_CAUSAL[env_name]
        kwargs = {}
        for i,key in enumerate(keys):
            kwargs[key] = env_info[i]
            
        kwargs['causal_hidden_dim'] = manager.model_parameters['causal_hidden_dim']
        kwargs['causal_dim'] = manager.model_parameters['causal_dim']
        return make_env(manager.model_parameters['env_name'], **kwargs)
    
    def encoder_env_info(self,env_info):
        return f'{env_info}'
