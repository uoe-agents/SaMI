import numpy as np
# import tensorflow as tf
from gym.envs.mujoco import mujoco_env
from gym import utils
import os

import gymnasium.spaces as spaces
from .utils import convert_observation_to_space


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self, mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25],causal_dim=-1,causal_hidden_dim=-1
    ):
        self.prev_pos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.causal_dim = causal_dim
        self.causal_hidden_dim = causal_hidden_dim
        self.current_trajectory_reward = 0
        self.current_trajectory_length = 0
        self.max_eps_length = 2000
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/humanoidstandup.xml" % dir_path, 5)

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
        super(HumanoidStandupEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation['observation'][None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def _get_obs(self):
        data = self.sim.data
        obs = {'observation':np.concatenate([data.qpos.flat[2:], data.qvel.flat]).astype(np.float32)}
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

    def step(self, a):
        # old_obs = np.copy(self._get_obs()['observation'])
        self.do_simulation(a, self.frame_skip)
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1
        alive_bonus = 1.0
        # lin_vel_cost = 0.25 / 0.015 * old_obs[..., 22]
        # quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        # quad_impact_cost = 0.0
        # qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        # alive_bonus = 5.0 * (1 - float(done))
        done = False
        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        self.current_trajectory_reward += reward
        self.current_trajectory_length += 1
        if self.current_trajectory_length == self.max_eps_length:
            return (
                self._get_obs(),
                reward,
                True,
                dict(
                    is_success=False,
                    episode=dict(
                        r = self.current_trajectory_reward,
                        l = self.current_trajectory_length
                    ),
                    reward_linvel=uph_cost,
                    reward_quadctrl=-quad_ctrl_cost,
                    reward_alive=alive_bonus,
                    reward_impact=-quad_impact_cost,
                ),
            )
        else:
            return (
                self._get_obs(),
                reward,
                False,
                dict(
                    is_success=False,
                    episode=dict(
                        r = self.current_trajectory_reward,
                        l = self.current_trajectory_length
                    ),
                    reward_linvel=uph_cost,
                    reward_quadctrl=-quad_ctrl_cost,
                    reward_alive=alive_bonus,
                    reward_impact=-quad_impact_cost,
                ),
            )

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-c, high=c, size=self.model.nv,),
        )
        pos_before = mass_center(self.model, self.sim)
        self.prev_pos = np.copy(pos_before)

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()

        return self._get_obs()

    def reward(self, obs, action, next_obs):
        # ctrl = action

        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1


        # lin_vel_cost = 0.25 / 0.015 * obs['observation'][..., 22]
        # quad_ctrl_cost = 0.1 * np.sum(np.square(ctrl), axis=-1)
        # quad_impact_cost = 0.0

        # done = bool((obs['observation'][..., 1] < 1.0) or (obs['observation'][..., 1] > 2.0))
        # alive_bonus = 5.0 * (not done)

        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            pos_after = self.sim.data.qpos[2]
            data = self.sim.data
            uph_cost = (pos_after - 0) / self.model.opt.timestep

            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1
            # ctrl = act

            # lin_vel_cost = 0.25 / 0.015 * obs[..., 22]
            # quad_ctrl_cost = 0.1 * tf.compat.v1.reduce_sum(tf.compat.v1.square(ctrl), axis=-1)
            # quad_impact_cost = 0.0

            # alive_bonus = 5.0 * tf.compat.v1.cast(
            #     tf.compat.v1.logical_and(tf.compat.v1.greater(obs[..., 1], 1.0), tf.compat.v1.less(obs[..., 1], 2.0)),
            #     dtype=tf.compat.v1.float32,
            # )

            # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            return reward

        return _thunk

    def change_mass(self, mass):
        self.mass_scale = mass

    def change_damping(self, damping):
        self.damping_scale = damping

    def change_env(self):
        mass = np.copy(self.original_mass)
        damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        damping *= self.damping_scale

        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])

    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return
    
    def reset(self, *, seed= None, options = None):
        self.current_trajectory_length = 0
        self.current_trajectory_reward = 0
        # return super().reset(seed=seed, options=options)
        return super().reset()
