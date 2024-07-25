import numpy as np
# import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env

import gymnasium.spaces as spaces
from .utils import convert_observation_to_space

class CrippleWalkerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, cripple_set=[0, 1, 2, 3], extreme_set=[0], mass_scale_set=[1.0], causal_dim=-1, causal_hidden_dim=-1):
        self.cripple_mask = None
        self.causal_dim = causal_dim
        self.causal_hidden_dim = causal_hidden_dim
        self.current_trajectory_reward = 0
        self.current_trajectory_length = 0
        self.max_eps_length = 2000
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)

        self.cripple_mask = np.ones(self.action_space.shape)
        self.cripple_set = cripple_set
        self.extreme_set = extreme_set

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.original_mass = np.copy(self.model.body_mass)
        self.mass_scale_set = mass_scale_set
        # self.damping_scale_set = damping_scale_set
        # self.original_damping = np.copy(self.model.dof_damping)

        utils.EzPickle.__init__(self, cripple_set, extreme_set)
        ob = self._get_obs()
        self.observation_space = convert_observation_to_space(ob)
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def _set_observation_space(self, observation):
        super(CrippleWalkerEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation['observation'][None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        if self.cripple_mask is None:
            a = a
        else:
            a = self.cripple_mask * a
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        terminated = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        done = False
        self.current_trajectory_reward += reward
        self.current_trajectory_length += 1
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
        obs = {'observation':np.concatenate(
            [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
        ).astype(np.float32)}
        if self.causal_dim > 0:
            obs['causal'] = np.random.randn(self.causal_dim).astype(np.float32)
            obs['hidden_h'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
            obs['hidden_c'] = np.zeros((self.causal_hidden_dim,),dtype=np.float32) # 这个时刻rnn的输出
        return obs

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        # random_index = self.np_random.randint(len(self.damping_scale_set))
        # self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()
        return self._get_obs()

    def reward(self, obs, action, next_obs):
        velocity = obs['observation'][..., 5]
        alive_bonus = 1.0
        reward = velocity
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum(axis=-1)
        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            velocity = obs[..., 5]
            alive_bonus = 1.0
            reward = velocity
            reward += alive_bonus
            reward -= 1e-3 * tf.compat.v1.reduce_sum(tf.compat.v1.square(act), axis=-1)
            return reward

        return _thunk

    def change_env(self):
        action_dim = self.action_space.shape
        self.cripple_mask = np.ones(action_dim)
        if self.extreme_set == [0]:
            self.crippled_joint = np.array([self.np_random.choice(self.cripple_set)])
            self.cripple_mask[self.crippled_joint] = 0
        elif self.extreme_set == [1]:
            self.crippled_joint = self.np_random.choice(self.cripple_set, 2, replace=False)
            self.cripple_mask[self.crippled_joint] = 0
        elif self.extreme_set == [2]:
            self.crippled_joint = np.array([])
        else:
            raise ValueError(self.extreme_set)
        
        geom_rgba = self._init_geom_rgba.copy()
        # f = open("./myfile.txt", "w")
        # f.writelines(str(geom_rgba))
        # f.writelines(str(self.model.joint_names))
        # f.writelines(str(self.model.geom_names))
        # f.writelines(str(self.crippled_joint))
        # f.writelines(str(self.model.geom_names.index('torso_geom')))
        # f.writelines(str(self.model.geom_names.index('thigh_geom')))
        # f.writelines(str(self.model.geom_names.index('leg_geom')))
        # f.writelines(str(self.model.geom_names.index('foot_geom')))
        # f.close()
        for joint in self.crippled_joint:
            geom_rgba[joint+2, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba.copy()

        mass = np.copy(self.original_mass)
        # damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        # damping *= self.damping_scale

        self.model.body_mass[:] = mass
        # self.model.dof_damping[:] = damping

    def change_mass(self, mass):
        self.mass_scale = mass

    # def change_damping(self, damping):
    #     self.damping_scale = damping

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 2
        # self.viewer.cam.distance = self.model.stat.extent * 0.75
        # self.viewer.cam.lookat[2] = 1.15
        # self.viewer.cam.elevation = -20
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def get_sim_parameters(self):
        return np.array([self.crippled_joint]).reshape(-1)

    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return

    def reset(self, *, seed= None, options= None):
        self.current_trajectory_length = 0
        self.current_trajectory_reward = 0
        # return super().reset(seed=seed, options=options)
        return super().reset()
