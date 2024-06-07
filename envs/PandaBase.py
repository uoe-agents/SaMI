#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   PandaBase.py
@Author  :   lixin 
@Version :   1.0
@Desc    :   None
'''

from typing import Any, Dict, Optional, Tuple
from abc import abstractmethod

import numpy as np

from panda_gym.envs.core import PyBulletRobot, RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks.pick_and_place import PickAndPlace

import gymnasium

class PandaBaseWrapper(PickAndPlace):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.11,
        obj_xy_range: float = 0.3,
        object_height: float= 1.0,
    ) -> None:
        self.object_height = object_height
        self.total_train_timesteps = 0
        self.cur_train_steps = 0
        super().__init__(sim,reward_type,distance_threshold,goal_xy_range,goal_z_range,obj_xy_range)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        object_size = np.ones(3) * self.object_size / 2
        object_size[2] = object_size[2] * self.object_height
        object_position = np.array([0.0, 0.0, object_size[2]])
        self.sim.create_box(
            body_name="object",
            half_extents=object_size,
            mass=1.0,
            position=object_position,
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=object_size,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    @abstractmethod
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        pass

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2 * self.object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def set_total_train_timesteps(self,total_train_timesteps):
        self.total_train_timesteps = total_train_timesteps

class PandaBaseEnv(RobotTaskEnv):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)

    def step(self, action):
        self.task.cur_train_steps += 1
        return super().step(action)

    def set_obs_friction_mass(self,friction,mass):
        self.friction = friction
        self.mass = mass

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = super()._get_obs()
        if self.causal_dim > 0 and self.causal_hidden_dim > 0:
            # obs['friction'] = np.log(self.friction + 0.001)
            # obs['mass'] = np.log(self.mass + 0.001)
            obs['causal'] = np.random.randn(self.causal_dim)
            obs['hidden_h'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
            obs['hidden_c'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
        return obs

