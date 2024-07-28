
import os
from gymnasium.envs.registration import register

# panda
from .PandaPickAndPlace import PandaPickAndPlaceEnv
from .PandaPush import PandaPushEnv
# domino
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

from .BaseHook import BaseHook
from .PandaHook import PandaHook
from .DominoHook import DominoHook

ENV_MAX_STEP = {
    'AntEnv': 2000,
    'CrippleHalfCheetahEnv':1000,
    'CrippleAntEnv':3000,
    'HalfCheetahEnv':1000,
    'SlimHumanoidEnv':1000,
    'HumanoidStandupEnv':2000,
    'HopperEnv':2000,
    'CrippleHopperEnv':1000,
    'WalkerEnv':2000,
    'CrippleWalkerEnv':2000,
    'WalkerHopperEnv':2000,
    'PandaPush-v3': 50
}

ENV_IDS = []
for task in ["Push", "PickAndPlace"]:
    env_id = f"Panda{task}-v3"
    register(
        id=env_id,
        entry_point=f"envs:Panda{task}Env",
        max_episode_steps=100 if task == "Stack" else 50,
    )

    ENV_IDS.append(env_id)

for env in ['AntEnv','CrippleHalfCheetahEnv','CrippleAntEnv','HalfCheetahEnv',
             'SlimHumanoidEnv','HumanoidStandupEnv','HopperEnv', 'CrippleHopperEnv', 'CrippleWalkerEnv', 'WalkerHopperEnv', 'WalkerEnv','Cartpoleenvs','Pendulumenvs']:
    env_id = f"{env}"
    register(
        id=env_id,
        entry_point=f"envs:{env}",
        max_episode_steps=2000,
        disable_env_checker=True
    )

    ENV_IDS.append(env_id)

__all__ = ['DominoHook','PandaHook','BaseHook']
