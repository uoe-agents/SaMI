import numpy as np
import os
# import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
import gymnasium.spaces as spaces
from .utils import convert_observation_to_space

class CrippleHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, cripple_set=[0, 1, 2, 3], extreme_set=[0], mass_scale_set=[1.0], causal_dim=-1, causal_hidden_dim=-1):
        """
        If extreme set=[0], neutral, cripple 1 joint
        If extreme set=[1], extreme, cripple 1 joints
        """
        self.prev_qpos = None

        self.cripple_mask = None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.causal_dim = causal_dim
        self.causal_hidden_dim = causal_hidden_dim
        self.current_trajectory_reward = 0
        self.current_trajectory_length = 0
        self.max_eps_length = 1000
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)

        self.cripple_mask = np.ones(self.action_space.shape)
        self.cripple_set = cripple_set
        self.extreme_set = extreme_set

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.original_mass = np.copy(self.model.body_mass)
        self.mass_scale_set = mass_scale_set

        utils.EzPickle.__init__(self, cripple_set, extreme_set)

        ob = self._get_obs()
        self.observation_space = convert_observation_to_space(ob)
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def _set_observation_space(self, observation):
        super(CrippleHalfCheetahEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation['observation'][None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        if self.cripple_mask is None:
            action = action
        else:
            action = self.cripple_mask * action
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1  * np.square(action).sum()
        reward_run = ob['observation'][0]
        reward = reward_run + reward_ctrl
        self.current_trajectory_reward += reward
        self.current_trajectory_length += 1
        done = False
        if self.current_trajectory_length == self.max_eps_length:
            return ob, reward, True, {'is_success':False,'episode':{'r':self.current_trajectory_reward,'l':self.current_trajectory_length}}
        else:
            return ob, reward, False, {'is_success':False,'episode':{'r':self.current_trajectory_reward,'l':self.current_trajectory_length}}

    def _get_obs(self):
        obs = {"observation":np.concatenate([
            (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ]).astype(np.float32)}
        if self.causal_dim > 0:
            obs['causal'] = np.random.randn(self.causal_dim).astype(np.float32)
            obs['hidden_h'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
            obs['hidden_c'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
        return obs

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        else:
            return tf.concat([obs[..., 1:2], tf.sin(obs[..., 2:3]), tf.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)

    def obs_postproc(self, obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)
        else:
            return tf.concat([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)

    def targ_proc(self, obs, next_obs):
        return np.concatenate([next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], axis=-1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        self.change_env()
        return self._get_obs()

    def reward(self, obs, action, next_obs):
        ctrl_cost = 1e-1 * np.sum(np.square(action), axis=-1)
        forward_reward = obs['observation'][..., 0]
        reward = forward_reward - ctrl_cost
        return reward
    
    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            ctrl_cost = 1e-1  * tf.reduce_sum(tf.square(act), axis=-1)
            forward_reward = obs[..., 0]
            reward = forward_reward - ctrl_cost
            return reward
        return _thunk
    
    def change_env(self):
        action_dim = self.action_space.shape
        if self.extreme_set == [0]:
            self.crippled_joint = np.array([self.np_random.choice(self.cripple_set)])
        elif self.extreme_set == [1]:
            self.crippled_joint = self.np_random.choice(self.cripple_set, 2, replace=False)
        else:
            raise ValueError(self.extreme_set)
        self.cripple_mask = np.ones(action_dim)
        self.cripple_mask[self.crippled_joint] = 0

        geom_rgba = self._init_geom_rgba.copy()
        for joint in self.crippled_joint:
            geom_idx = self.model.geom_names.index(self.model.joint_names[joint+3])
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba.copy()

        mass = np.copy(self.original_mass)
        mass *= self.mass_scale
        self.model.body_mass[:] = mass

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
    
    def get_sim_parameters(self):
        return np.array([self.crippled_joint]).reshape(-1)
    
    def num_modifiable_parameters(self):
        return 1

    def log_diagnostics(self, paths, prefix):
        return

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def reset(self, *, seed= None, options = None):
        self.current_trajectory_length = 0
        self.current_trajectory_reward = 0
        # return super().reset(seed=seed, options=options)
        return super().reset()