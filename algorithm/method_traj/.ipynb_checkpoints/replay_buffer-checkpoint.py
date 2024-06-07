# -*- coding:utf-8 -*-

import warnings
from typing import Any, Dict, Generator, List, Optional, Union
import random

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize
# data define
from line_profiler import profile

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union
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
    pos_trajectory_dones: ListTensor
    neg_trajectories: ListTensorDict
    neg_trajectory_rewards: ListTensor
    neg_trajectory_dones: ListTensor


# 细粒度对比学习采样
class ContrastBufferSamplesFineGrained(NamedTuple):
    trajectories: ListTensorDict #  
    trajectory_dones: ListTensor
    pos_trajectory_idx: ListTensor
    neg_trajectory_idx: ListTensor
    pos_trajectory_rewards: ListTensor
    neg_trajectory_rewards: ListTensor

class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

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

        causal_keys:set = None,
        buffer_type:str = "all", # {'all','one','half','env_diff'}
        max_eps_length:int = 50,
        neg_radio:float=0.5,
        fine_grained_frame_stack=10,# 细粒度采样窗口大小
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        # Adjust buffer size
        self.buffer_size = buffer_size
        self.neg_radio = neg_radio
        self.fine_grained_frame_stack = fine_grained_frame_stack

        self.observations = {
            key: np.random.rand(
                    self.buffer_size, max_eps_length + 1, self.n_envs, *_obs_shape
                ).astype(observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
            if key not in {'causal', 'hidden_c','hidden_h'}
        }

        self.actions = np.random.rand(
            self.buffer_size, max_eps_length + 1, self.n_envs, self.action_dim
        ).astype(self._maybe_cast_dtype(action_space.dtype))
        self.rewards = np.random.rand(self.buffer_size, max_eps_length + 1, self.n_envs).astype(np.float32)
        self.dones = np.zeros((self.buffer_size, max_eps_length + 1, self.n_envs)).astype(np.float32)

        self.traj_len = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.average_return = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.actions[:,-1] = 0

        self.buffer_type = buffer_type
        self.max_eps_length = max_eps_length
        self.step_pos = np.zeros(self.n_envs, dtype=np.int32)
        self.traj_pos = np.zeros(self.n_envs, dtype=np.int32)
        self.full = [False for _ in range(self.n_envs)]

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
        env_ids = np.arange(self.n_envs)
        for key in self.observations.keys():
            self.observations[key][self.traj_pos, self.step_pos, env_ids] = np.array(obs[key])

        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.traj_pos, self.step_pos, env_ids] = np.array(action)
        self.rewards[self.traj_pos, self.step_pos, env_ids] = np.array(reward)
        self.dones[self.traj_pos, self.step_pos, env_ids] = np.array(done)
        self.step_pos = self.step_pos + 1

        for i, _done in enumerate(done):
            if _done:
                self.average_return[self.traj_pos[i], i] = np.sum(self.rewards[self.traj_pos[i],:self.step_pos[i] + 1,i]) / (self.step_pos[i] + 1)
                self.dones[self.traj_pos[i], self.step_pos[i]:, i] = 1
                self.traj_len[self.traj_pos[i], i] = self.step_pos[i]
                self.step_pos[i] = 0
                self.traj_pos[i] += 1
                if self.traj_pos[i] == self.buffer_size:
                    self.full[i] = True
                    self.traj_pos[i] = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        get rl_samples [batch_size, feature_size] and rnn_samples [batch_size, seq_len, feature_size]
        rnn_samples[:, -1] == rl_samples

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: 
        """

        env_ids = np.arange(self.n_envs).repeat(batch_size//self.n_envs)
        batch_inds = []
        for _env_id in range(self.n_envs):
            upper_bound = self.buffer_size if self.full[_env_id] else self.traj_pos[_env_id]
            batch_inds.append(np.random.randint(0, upper_bound, size=batch_size//self.n_envs))
        batch_inds = np.concatenate(batch_inds,axis=0)
        return self._get_samples(batch_inds, env_ids, env=env)
    
    def _sample_eps_infos(self, batch_size):

        if self.buffer_type == 'all':
            env_ids = np.arange(self.n_envs).repeat(batch_size//self.n_envs)
            batch_inds = []
            for _env_id in range(self.n_envs):
                upper_bound = self.buffer_size if self.full[_env_id] else self.traj_pos[_env_id]
                batch_inds.append(np.random.randint(0, upper_bound, size=batch_size//self.n_envs))
            batch_inds = np.concatenate(batch_inds,axis=0)
            rewards = self.average_return[batch_inds, env_ids]

            eps_info= np.stack((batch_inds, env_ids),axis=1)
            eps_info= eps_info[rewards.argsort()]
            # Ensure that both positive and negative examples have samples
            neg_num = np.clip(int(batch_size * self.neg_radio), 1, batch_size-1)
            return [eps_info[neg_num:]],[eps_info[:neg_num]]
        elif self.buffer_type == 'one':
            env_id = np.random.randint(0, self.n_envs)
            env_ids = np.repeat(env_id, batch_size)
            upper_bound = self.buffer_size if self.full[env_id] else self.traj_pos[env_id]
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            rewards = self.average_return[batch_inds, env_ids]
            eps_info = np.stack((batch_inds, env_ids),axis=1)
            eps_info = eps_info[rewards.argsort()]
            neg_num = np.clip(int(batch_size * self.neg_radio), 1, batch_size-1)
            return [eps_info[neg_num:]],[eps_info[:neg_num]]
        elif self.buffer_type == 'half':
            neg_num = np.clip(int(batch_size * self.neg_radio), 1, batch_size-1)
            env_ids = np.arange(self.n_envs).repeat(batch_size//self.n_envs)
            batch_inds = []
            for _env_id in range(self.n_envs):
                upper_bound = self.buffer_size if self.full[_env_id] else self.traj_pos[_env_id]
                batch_inds.append(np.random.randint(0, upper_bound, size=batch_size//self.n_envs))
            batch_inds = np.concatenate(batch_inds,axis=0)
            rewards = self.average_return[batch_inds, env_ids]

            eps_info= np.stack((batch_inds, env_ids),axis=1)
            eps_info= eps_info[rewards.argsort()]

            neg_eps_infos = [eps_info[:neg_num]]
            neg_reward = self.average_return[neg_eps_infos[0][-1,0],neg_eps_infos[0][-1,1]]
            pos_eps_infos = []
            # find reward bigger than neg[-1]
            other_env_ids = [env_i for env_i in range(self.n_envs)]
            random.shuffle(other_env_ids)
            while len(pos_eps_infos) == 0:
                # exist other_env_id
                env_id = other_env_ids[-1]
                other_env_ids.pop()
                upper_bound = self.buffer_size if self.full[env_id] else self.traj_pos[env_id]
                pos_batch_inds = np.where(self.average_return[:upper_bound,env_id] > neg_reward)[0]
                if pos_batch_inds.shape[0] > 0:
                    pos_env_ids = np.repeat(env_id, batch_size-neg_num)
                    if batch_size-neg_num <= pos_batch_inds.shape[0]:
                        pos_batch_inds = np.random.choice(pos_batch_inds, batch_size-neg_num, replace=False)
                    else:
                        pos_batch_inds = np.random.choice(pos_batch_inds, batch_size-neg_num, replace=True)

                    pos_eps_infos.append(np.stack((pos_batch_inds, pos_env_ids), axis=1))

            return pos_eps_infos, neg_eps_infos

        # elif self.buffer_type == 'fine_grained':
        #     neg_num = np.clip(int(batch_size * self.neg_radio), 1, batch_size-1)
        #     batch_inds = []
        #     for _env_id in range(self.n_envs):
        #         upper_bound = self.buffer_size if self.full[_env_id] else self.traj_pos[_env_id]
        #         batch_inds.append(np.random.randint(0, upper_bound, size=batch_size//self.n_envs))
        #     batch_inds = np.concatenate(batch_inds,axis=0)
        #     env_ids = np.arange(self.n_envs).repeat(batch_size//self.n_envs)
        #     rewards = self.rewards[batch_inds, :, env_ids]

        elif self.buffer_type == 'env_diff':
            neg_num = np.clip(int(batch_size * self.neg_radio), 1, batch_size-1)
            env_i = np.random.randint(0, self.n_envs)
            other_env_i = random.choice([_env_i for _env_i in range(self.n_envs) if _env_i != env_i])
            env_ids = np.repeat(env_i, batch_size-neg_num)
            upper_bound = self.buffer_size if self.full[env_i] else self.traj_pos[env_i]
            batch_inds = np.random.randint(0, upper_bound, size=batch_size-neg_num)
            other_env_ids = np.repeat(other_env_i ,neg_num)
            other_upper_bound = self.buffer_size if self.full[other_env_i] else self.traj_pos[other_env_i]
            other_batch_inds = np.random.randint(0, other_upper_bound, size=neg_num)
            return [np.stack((batch_inds, env_ids), axis=1)], [np.stack((other_batch_inds, other_env_ids), axis=1)]
        else:
            raise NotImplementedError

    def _to_trajectory(self, eps_infos):
        # eps_infos[:,0] batch_ids
        # eps_infos[:,1] env_ids
        batch_inds = eps_infos[:, 0]
        env_inds = eps_infos[:, 1]
        trajectory = {}
        
        for key in self.observations:
            trajectory[key] = self.observations[key][batch_inds,:,env_inds]
        trajectory = self._normalize_obs(trajectory)
        trajectory['action'] = self.actions[batch_inds-1,:,env_inds]

        return trajectory

    def sample_contrast(self, batch_size: int) -> ContrastBufferSamples:
        """
        Sample Contrast
        """

        pos_eps_infos, neg_eps_infos = self._sample_eps_infos(batch_size) # [np.array(traj_id, env_id)],[np.array]
        pos_trajectories,pos_trajectory_rewards,pos_trajectory_dones = [],[],[]
        neg_trajectories,neg_trajectory_rewards,neg_trajectory_dones = [],[],[]
        for _pos_eps_info, _neg_eps_info in zip(pos_eps_infos, neg_eps_infos):
            if self.buffer_type != 'env_diff':
                pos_trajectory_rewards.append(self.to_torch(self.average_return[_pos_eps_info[:,0], _pos_eps_info[:,1]]))
                neg_trajectory_rewards.append(self.to_torch(self.average_return[_neg_eps_info[:,0], _neg_eps_info[:,1]]))
            else:
                pos_trajectory_rewards.append(self.to_torch(1.0))
                neg_trajectory_rewards.append(self.to_torch(0.0))

            pos_dones =self.dones[_pos_eps_info[:,0],:,_pos_eps_info[:,1]]
            neg_dones =self.dones[_neg_eps_info[:,0],:,_neg_eps_info[:,1]]

            pos_trajectory_dones.append(self.to_torch(pos_dones))
            neg_trajectory_dones.append(self.to_torch(neg_dones))

            _pos_eps_info = self._to_trajectory(_pos_eps_info)
            _neg_eps_info = self._to_trajectory(_neg_eps_info)
            for key in _pos_eps_info:
                _pos_eps_info[key] = self.to_torch(_pos_eps_info[key])
                _neg_eps_info[key] = self.to_torch(_neg_eps_info[key])

            pos_trajectories.append(_pos_eps_info)
            neg_trajectories.append(_neg_eps_info)
        
        return ContrastBufferSamples(
            pos_trajectories=pos_trajectories,
            pos_trajectory_rewards=pos_trajectory_rewards,
            pos_trajectory_dones=pos_trajectory_dones,
            neg_trajectories=neg_trajectories,
            neg_trajectory_rewards=neg_trajectory_rewards,
            neg_trajectory_dones=neg_trajectory_dones
        )

    def _get_samples(self, batch_inds: np.ndarray, env_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        
        observations = {key: obs[batch_inds,:,env_inds] for key, obs in self.observations.items()}
        observations = self._normalize_obs(observations)
        observations['action'] = np.concatenate((np.zeros((batch_inds.shape[0], 1, self.actions.shape[-1]),dtype=np.float32),self.actions[batch_inds,:-1,env_inds]),axis=1)
        observations = {key: self.to_torch(obs) for key, obs in observations.items()}

        actions = self.to_torch(self.actions[batch_inds,:-1, env_inds])
        dones = self.to_torch(self.dones[batch_inds,:-1, env_inds]).unsqueeze(-1)
        rewards = self.to_torch(self._normalize_reward(self.rewards[batch_inds,:-1, env_inds], env)).unsqueeze(-1)

        replay_data = DictReplayBufferSamples(
            observations= observations,
            actions=actions,
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

