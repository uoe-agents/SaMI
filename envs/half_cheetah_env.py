import numpy as np
import os
# import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
import gymnasium.spaces as spaces
from .utils import convert_observation_to_space

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self, mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25], causal_dim=-1,causal_hidden_dim=-1
    ):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.causal_dim = causal_dim
        self.causal_hidden_dim = causal_hidden_dim
        self.current_trajectory_reward = 0
        self.current_trajectory_length = 0
        self.max_eps_length = 1000
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/half_cheetah.xml" % dir_path, 5)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set)
        ob = self._get_obs()
        self.observation_space = convert_observation_to_space(ob)
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def _set_observation_space(self, observation):
        super(HalfCheetahEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation['observation'][None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob['observation'][0]
        reward = reward_run + reward_ctrl
        self.current_trajectory_reward += reward
        self.current_trajectory_length += 1
        done = False
        if self.current_trajectory_length == self.max_eps_length:
            return ob, reward, True, {'is_success':False,'episode':{'r':self.current_trajectory_reward,'l':self.current_trajectory_length}}
        else:
            return ob, reward, False, {'is_success':False,'episode':{'r':self.current_trajectory_reward,'l':self.current_trajectory_length}}

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def _get_obs(self):
        obs = {'observation': np.concatenate(
            [
                (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        ).astype(np.float32)}
        if self.causal_dim > 0:
            obs['causal'] = np.random.randn(self.causal_dim).astype(np.float32)
            obs['hidden_h'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
            obs['hidden_c'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
        return obs

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate(
                [
                    obs[..., 1:2],
                    np.sin(obs[..., 2:3]),
                    np.cos(obs[..., 2:3]),
                    obs[..., 3:],
                ],
                axis=-1,
            )
        else:
            return tf.compat.v1.concat(
                [
                    obs[..., 1:2],
                    tf.compat.v1.sin(obs[..., 2:3]),
                    tf.compat.v1.cos(obs[..., 2:3]),
                    obs[..., 3:],
                ],
                axis=-1,
            )

    def obs_postproc(self, obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate(
                [pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1
            )
        else:
            return tf.compat.v1.concat([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)

    def targ_proc(self, obs, next_obs):
        return np.concatenate(
            [next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], axis=-1
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(
            loc=0, scale=0.001, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.normal(
            loc=0, scale=0.001, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()
        return self._get_obs()

    def reward(self, obs, action, next_obs):
        ctrl_cost = 1e-1 * np.sum(np.square(action), axis=-1)
        forward_reward = obs[..., 0]
        reward = forward_reward - ctrl_cost
        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            ctrl_cost = 1e-1 * tf.compat.v1.reduce_sum(tf.compat.v1.square(act), axis=-1)
            forward_reward = obs[..., 0]
            reward = forward_reward - ctrl_cost
            return reward

        return _thunk

    def change_env(self):
        mass = np.copy(self.original_mass)
        damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        damping *= self.damping_scale

        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping

    def change_mass(self, mass):
        self.mass_scale = mass

    def change_damping(self, damping):
        self.damping_scale = damping

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

    def get_sim_parameters(self):
        training_mass_set = [0.25, 0.5, 1.5, 2.5]
        mass_avg = np.mean(training_mass_set)
        mass_std = np.std(training_mass_set)
        normalized_mass = (self.mass_scale - mass_avg) / mass_std
        return np.array([normalized_mass, self.damping_scale])

    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return

    def reset(self, *, seed= None, options= None):
        self.current_trajectory_length = 0
        self.current_trajectory_reward = 0
        # return super().reset(seed=seed, options=options)
        return super().reset()