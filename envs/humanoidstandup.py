import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

import gymnasium.spaces as spaces
from .utils import convert_observation_to_space

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, **kwargs):
        observation_space = Box(
            self, mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25],causal_dim=-1,causal_hidden_dim=-1
        )
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.causal_dim = causal_dim
        self.causal_hidden_dim = causal_hidden_dim
        self.current_trajectory_reward = 0
        self.current_trajectory_length = 0
        self.max_eps_length = 2000
        mujoco_env.MujocoEnv.__init__(
            self,
            "%s/assets/humanoidstandup.xml" % dir_path,
            5,
            observation_space=observation_space,
            **kwargs
        )
        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set
        utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        if self.render_mode == "human":
            self.render()
        return (
            self._get_obs(),
            reward,
            False,
            False,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20
