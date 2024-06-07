# -*- coding:utf-8 -*-

import warnings
from typing import Any, Dict, Generator, List, Optional, Union
import random

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
# data define
from line_profiler import profile

from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union
TensorDict = Dict[str, th.Tensor]
ListTensorDict = List[TensorDict]
ListTensor = List[th.Tensor]

# Avoid circular imports, we use type hint as string to avoid it too
if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

# 多组对比学习采样
class ContrastBufferSamples(NamedTuple):
    pos_trajectories: ListTensorDict #  
    pos_trajectory_rewards: ListTensor
    neg_trajectories: ListTensorDict
    neg_trajectory_rewards: ListTensor

class DictReplayBuffer(BaseBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        causal_keys:set = None,
        frame_stack:int = 2,
        contrast_frame_stack = 10,
        buffer_type:str = "all", # {'all','one','half','env_diff'}
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        # Adjust buffer size
        self.buffer_size = buffer_size

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.random.rand(self.buffer_size, self.n_envs, *_obs_shape).astype(observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.random.rand(self.buffer_size, self.n_envs, *_obs_shape).astype(observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.random.rand(
            self.buffer_size, self.n_envs, self.action_dim
        ).astype(self._maybe_cast_dtype(action_space.dtype))
        self.rewards = np.random.rand(self.buffer_size, self.n_envs).astype(np.float32)
        self.dones = np.random.rand(self.buffer_size, self.n_envs).astype(np.float32)

        self.buffer_type = buffer_type

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.frame_stack = frame_stack # 采样返回连续step数量
        self.contrast_frame_stack = contrast_frame_stack

        if causal_keys is None:
            self.causal_keys = set()
        else:
            self.causal_keys = causal_keys

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        get rl_samples [batch_size, feature_size] and rnn_samples [batch_size, seq_len, feature_size]
        rnn_samples[:, -1] == rl_samples

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: 
        """

        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(self.frame_stack, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _to_trajectory(self, eps_infos):
        # batch_size, frame_stack + 1(env_id)]
        traj_i = eps_infos[:, :-1]
        envs = eps_infos[:, -1]

        batch_size = traj_i.shape[0]
        traj_i = traj_i.reshape(-1)
        _envs = np.repeat(envs, self.contrast_frame_stack)

        trajectory = {} # action, observation, hidden_c, hidden_h
        dones = self.dones[traj_i-1,_envs].reshape(batch_size,self.contrast_frame_stack)
        # action_{t-1} + observation_{t} + hidden_state_{t-1} -> causal_{t} / hidden_state_{t}
        trajectory['action'] = self.actions[traj_i-1,_envs].reshape(batch_size,self.contrast_frame_stack, -1)
        trajectory['action'] = trajectory['action'] * (1-dones[:,:,None])
        for key in self.observations:
            if key == 'hidden_c' or key == 'hidden_h':
                trajectory[key] = self.observations[key][traj_i-1,_envs].reshape(batch_size,self.contrast_frame_stack,-1)
            if key not in self.causal_keys and key != 'causal':
                trajectory[key] = self.observations[key][traj_i,_envs].reshape(batch_size,self.contrast_frame_stack, -1)

        return trajectory

    def _sample_eps_infos(self, env_i, batch_size):
        """
        采样轨迹 获取开始和结束下标
        """

        if self.full:
            max_pos_idx = self.buffer_size
        else:
            max_pos_idx = self.pos
 
        if self.buffer_type == 'one':
            env_ids = np.ones(batch_size, dtype=np.int32) * env_i
        elif self.buffer_type == 'all' or self.buffer_type == 'half':
            env_ids = np.random.randint(0, self.n_envs, size=(batch_size,))
        elif self.buffer_type == 'env_diff':
            env_ids_1 = np.ones(batch_size//2, dtype=np.int32) * env_i
            other_envs = [i for i in range(self.n_envs) if i != env_i]
            env_ids_2 = np.random.choice(other_envs, size=(batch_size//2))
            env_ids = np.concatenate((env_ids_2, env_ids_1),axis=0) # pos put last
        else:
            raise NotImplementedError

        eps_start_ids = np.random.randint(0, max_pos_idx-self.contrast_frame_stack, size=(batch_size,))

        eps_ids = []
        for offset in range(self.contrast_frame_stack):
            eps_ids.append((eps_start_ids + offset) % self.buffer_size)
        eps_ids = np.stack(eps_ids, axis=1).reshape(-1)
        _env_ids = np.repeat(env_ids, self.contrast_frame_stack)
        # This is just a way to give trajectory rewards
        rewards = self.rewards[eps_ids, _env_ids].reshape(-1, self.contrast_frame_stack).mean(axis = 1, keepdims=True)
        eps_infos = np.concatenate((eps_ids.reshape(-1, self.contrast_frame_stack), env_ids[:,None], rewards),axis=1)
        return eps_infos

    def binary_search(self, eps_infos):
        env_l,env_r = eps_infos[0,-2], eps_infos[-1,-2]
        l,r = 0,eps_infos.shape[0]-1
        ans = []
        for env_i in range(int(env_l),int(env_r)):
            while l < r:
                m = (l + r) // 2
                if eps_infos[m, -2] > env_i:
                    r = m
                else:
                    l = m + 1
            r = eps_infos.shape[0]-1
            if len(ans) == 0 or ans[-1] != l:
                ans.append(l)
        return ans

    def sample_contrast(self, batch_size: int) -> ContrastBufferSamples:
        """
        Sample batch_size//2 pos, and batch_size//2 neg
        every env average
        """
        micro_batch_size = batch_size // 2 // self.n_envs

        pos_trajectories = []
        neg_trajectories = []

        pos_trajectory_rewards = []
        neg_trajectory_rewards = []

        for env_i in range(self.n_envs):
            _eps_infos = self._sample_eps_infos(env_i, int(micro_batch_size * 2))
            if self.buffer_type != 'env_diff':
                _eps_infos = _eps_infos[_eps_infos[:, -1].argsort()]

            _pos_eps_infos = [_eps_infos[_eps_infos.shape[0]//2:]]
            _neg_eps_infos = _eps_infos[:_eps_infos.shape[0]//2]

            _neg_trajectory_rewards = self.to_torch(_neg_eps_infos[:,-1].astype(np.float32))
            _neg_trajectories = self._to_trajectory(_neg_eps_infos[:, :-1].astype(np.int32))
            for key in _neg_trajectories: 
                _neg_trajectories[key] = self.to_torch(_neg_trajectories[key])

            if self.buffer_type == 'half':
                # 
                origin_pos_eps_infos = _pos_eps_infos[0]
                origin_pos_eps_infos = origin_pos_eps_infos[origin_pos_eps_infos[:, -2].argsort()]
                split_ids = self.binary_search(origin_pos_eps_infos)
                _pos_eps_infos = []
                split_ids = [0] + split_ids + [origin_pos_eps_infos.shape[0]]

                for start_id, end_id in zip(split_ids[:-1], split_ids[1:]):
                    _pos_eps_infos.append(origin_pos_eps_infos[start_id:end_id])

            for _pos_eps_infos_i in _pos_eps_infos:

                neg_trajectory_rewards.append(_neg_trajectory_rewards)
                neg_trajectories.append(_neg_trajectories)

                _pos_trajectories = self._to_trajectory(_pos_eps_infos_i[:, :-1].astype(np.int32))
                for key in _pos_trajectories:
                    _pos_trajectories[key] = self.to_torch(_pos_trajectories[key])
                pos_trajectory_rewards.append(self.to_torch(_pos_eps_infos_i[:,-1].astype(np.float32)))
                pos_trajectories.append(_pos_trajectories)

        return ContrastBufferSamples(
            pos_trajectories=pos_trajectories,
            pos_trajectory_rewards=pos_trajectory_rewards,
            neg_trajectories=neg_trajectories,
            neg_trajectory_rewards=neg_trajectory_rewards,
        )

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        
        observations = {key: obs[batch_inds,env_indices] for key, obs in self.observations.items()}
        next_observations = {key: obs[batch_inds,env_indices] for key, obs in self.next_observations.items()}

        observations['action'] = self.actions[batch_inds-1,env_indices] * (1 - self.dones[batch_inds-1,env_indices][:,None])
        next_observations['action'] = self.actions[batch_inds,env_indices] * (1 - self.dones[batch_inds,env_indices][:,None])
        observations['hidden_c'] = self.observations['hidden_c'][batch_inds-1,env_indices]
        observations['hidden_h'] = self.observations['hidden_h'][batch_inds-1,env_indices]
        next_observations['hidden_c'] = self.next_observations['hidden_c'][batch_inds-1,env_indices]
        next_observations['hidden_h'] = self.next_observations['hidden_h'][batch_inds-1,env_indices]

        observations = self._normalize_obs(observations)
        next_observations = self._normalize_obs(next_observations)
        observations = {key: self.to_torch(obs) for key, obs in observations.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_observations.items()}

        actions = self.to_torch(self.actions[batch_inds, env_indices])
        dones = self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)
        rewards = self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env))

        replay_data = DictReplayBufferSamples(
            observations= observations,
            actions= actions,
            next_observations= next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=dones,
            rewards=rewards,
        )
        return replay_data

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

