# Skill-aware Mutural Information (SaMI)

This is the official implementation of Skill-aware Mutural Information (SaMI) from the paper Skill-aware Mutual Information Optimisation for Generalisation in Reinforcement Learning. on arxiv: https://arxiv.org/pdf/2406.04815

by [Xuehui Yu](https://github.com/yuxuehui), [Mhairi Dunion](https://github.com/mhairidunion), [Xin Li](https://github.com/loxs123), Stefano V. Albrecht

**Challenge Problem: **
Meta-Reinforcement Learning (Meta-RL) agents can struggle to operate across tasks with varying environmental features that require different optimal skills (i.e., different modes of behaviors). An ideal RL algorithm should be able to learn a single policy to perform multiple tasks and generalise to new and unseen tasks.

https://github.com/user-attachments/assets/2d644ccb-851c-4053-bfc4-0923f5c26080

SaMI is a plug-and-play module that can be integrated with any Meta-RL algorithm. In this repository, we provide implementations of two baselines: [CCM](https://cdn.aaai.org/ojs/16914/16914-13-20408-1-2-20210518.pdf) and [TESAC](https://arxiv.org/pdf/1910.10897). We have equipped these baselines with SaMI, resulting in SaCCM and SaTESAC.

**TODO**

- [ ] Change name "SaSAC" to "SaTESAC";
- [ ] Change name "DominoHook" to "MujocoHook"
- [ ] Change name "train_env" to "train_task"
- [ ] Change name "test_env" to "test_task"
- [ ] Main part needs videos which show SaCCM can Push, Pick&Place
- [ ] The number of training tasks must be a multiple of 4, otherwise there will be a tensor dimension mismatch issue.

### 🤖 🤖 Our Skill-aware Robot 👏 👏

#### Panda-gym ✔️

We open-sourced our modified Panda-gym benchmark on [Skill-aware-Panda-gym](https://github.com/uoe-agents/Skill-aware-Panda-gym). During the training process, we alter the values of cube mass and table friction. In the test setting, we test on previously unseen mass and friction values. The following shows the training and testing videos of SaCCM on different tasks (0.5x speed):

<table cellpadding="3" cellspacing="0" style="width: 60%;margin:auto">
    <tr>
        <td width="18%" height="100%">
            Friction = 5.0
        </td>
        <td width="20%" height="100%">
            <img src="docs/gif/panda_gym/SaCCM-frction5-mass1.gif" width=200>
        </td>
	<td width="20%" height="100%">
            <img src="docs/gif/panda_gym/SaCCM_friction5_mass5.gif" width=200>
        </td>
	<td width="20%" height="100%">
        </td>
    </tr>
    <tr>
        <td width="18%" height="100%">
            Friction = 1.0
        </td>
        <td width="20%" height="100%">
            <img src="docs/gif/panda_gym/SaCCM-frction1-mass1.gif" width=200>
        </td>
	<td width="20%" height="100%">
            <img src="docs/gif/panda_gym/SaCCM-friction1-mass5.gif" width=200>
        </td>
	<td width="20%" height="100%">
		<img src="docs/gif/panda_gym/SaCCM_friction1_mass10.gif" width=200>
        </td>
    </tr>
	<tr>
        <td width="18%" height="100%">
            Friction = 0.1
        </td>
        <td width="20%" height="100%">
            <img src="docs/gif/panda_gym/SaCCM_friction0.1_mass1.0.gif" width=200>
        </td>
	<td width="20%" height="100%">
            <img src="docs/gif/panda_gym/SaCCM-friction0.1-mass5.gif" width=200>
        </td>
	<td width="20%" height="100%">
		<img src="docs/gif/panda_gym/SaCCM-friction0.1-mass30.gif" width=200>
        </td>
    </tr>
<tr>
        <td width="18%" height="100%"></td>
        <td width="20%" height="100%">Mass = 1.0</td>
	<td width="20%" height="100%">Mass = 5.0 </td>
	<td width="20%" height="100%">Mass = 10.0</td>
    </tr>
</table>

#### Crippled Half-Cheetah ✔️

During the training process, we alter the values of mass and damping, and randomly cripple ***one of the front leg joints*** (joint 3, 4 or 5). In the test setting, we test on previously unseen mass and damping values. More importantly, in the moderate test setting, we randomly cripple ***one of the back leg joints*** (joint 0, 1 or 2). In the extreme test setting, we increase the difficulty by randomly crippling ***two joints from both the front and back legs***. We found that SaCCM and SaTESAC can adjust the running posture of the Half-Cheetah to cope with different crippled joints, thereby running as far as possible using different modes of running (i.e., different skills). The following shows the training and testing videos of SaCCM on different tasks (first 60 seconds, 2x speed):

<p align=center>
<img src="docs/gif/crippledhalfcheetah/([3], [0], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([0], [0], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([1], [0], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([2], [0], [1.0]).gif" width=200>
</p>
<p align=center>
<img src="docs/gif/crippledhalfcheetah/([0, 3], [1], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([1, 4], [1], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([4, 5], [1], [1.0]).gif" width=200>
</p>

And some faliure cases of CCM:

<p align=center>
<img src="docs/gif/crippledhalfcheetah/CCM_Training Test (Cripple Rotor 3).gif" width=200> <img src="docs/gif/crippledhalfcheetah/CCM_Moderate Test (Cripple Rotor 0).gif" width=200> <img src="docs/gif/crippledhalfcheetah/CCM_Moderate Test (Cripple Rotor 1).gif" width=200> <img src="docs/gif/crippledhalfcheetah/CCM_Moderate Test (Cripple Rotor 2).gif" width=200>
</p>
<p align=center>
<img src="docs/gif/crippledhalfcheetah/CCM_Extreme Test (Cripple Rotor 0&3).gif" width=200> <img src="docs/gif/crippledhalfcheetah/CCM_Extreme Test (Cripple Rotor 1&4).gif" width=200> <img src="docs/gif/crippledhalfcheetah/CCM_Extreme Test (Cripple Rotor 4&5).gif" width=200>
</p>

#### Crippled Ant

During the training process, we alter the values of mass and damping, and randomly cripple one of legs (leg 0/1/2/3). In the test setting, we test on previously unseen mass and damping values. More importantly, in the extreme test setting, we also randomly cripple two legs. The following shows the training and testing videos of SaCCM on different tasks (first 60 seconds, 2x speed):

<p align=center>
<img src="docs/gif/crippledant/SaCCM_Training_(Crippled_0).gif" width=200> <img src="docs/gif/crippledant/SaCCM_Training_(Crippled_1&4).gif" width=200>
</p>

* [ ] SAC single-task training setting: [([0], [0],[1.0])], reward ctrl = 0.1, 0.05
* [ ] SAC single-task training setting: [([0], [2],[1.0])], reward ctrl = 0.1, 0.05
* [ ] SAC single-task training setting: [([0,1], [1],[1.0])], reward ctrl = 0.1, 0.05s
* [ ] SAC single-task training setting: [([0,3], [1],[1.0])], reward ctrl = 0.1, 0.05

#### Hopper ✔️

During the training process, we alter the values of mass and damping. In the test setting, we test on previously unseen mass and damping values. In the Hopper environment, we found that in order to adapt to different mass values, the TESAC/CCM/SaTESAC/SaCCM policy tends to learn only one skill, which is ***the Hopper hopping forward on the floor***. The following shows the training and testing videos of SaTESAC on different tasks (first 180 seconds, 6x speed):

<p align=center>
<img src="docs/gif/hopper/Extreme_Test_(mass=0.25, damping=0.25).gif" width=200> <img src="docs/gif/hopper/SaTESAC_Extreme_Test_(mass=0.25, damping=1.75).gif" width=200> <img src="docs/gif/hopper/CCM_Extreme_Test_(mass=1.75, damping=0.25).gif" width=200> <img src="docs/gif/hopper/CCM_Extreme_Test_(mass=1.75, damping=1.75).gif" width=200> <img src="docs/gif/hopper/SaTESAC_Super_Extreme_Test_(mass=4.0, damping=1.0).gif" width=200> 
</p>

#### Walker ✔️

During the training process, we alter the values of mass and damping. In the test setting, we test on previously unseen mass and damping values. In the Walker environment, we found that in order to adapt to different mass values, the TESAC/CCM/SaTESAC/SaCCM policy tends to learn only one skill, which is ***the Walker hopping forward on the floor***. The following shows the training and testing videos of SaTESAC on different tasks (first 180 seconds, 6x speed):

<p align=center>
<img src="docs/gif/walker/walker-sasac-0.250.25.gif" width=200> <img src="docs/gif/walker/walker-sasac-0.251.75.gif" width=200> <img src="docs/gif/walker/walker-sasac-1.750.25.gif" width=200> <img src="docs/gif/walker/walker-sasac-1.751.75.gif" width=200> <img src="docs/gif/walker/walker-sasac-4.01.0.gif" width=200> 
</p>

* [ ] multiple seeds multi-task training (same to Hopper)

#### WalkerHopper ✔️

During the training process, we alter the values of mass and damping, and randomly disable the right leg (i.e., three joints 0, 1, and 2). In the test setting, we evaluate on previously unseen mass and damping values. More importantly, in the moderate test setting, we randomly disable one joint of the right leg (i.e., joints 0, 1, or 2); in the extreme test setting, we also randomly disable the left leg (i.e., three joints 3, 4, and 5) or one joint of the left leg (i.e., joint 3, 4, or 5). The following shows the training and testing videos of SaCCM on different tasks (first 180 seconds, 6x speed):

<p align=center>
<img src="docs/gif/walkerhopper/SaCCM_no_crippled_joints.gif" width=200> <img src="docs/gif/walkerhopper/SaCCM_([0, 1, 2], [3], [1.0]).gif" width=200> <img src="docs/gif/walkerhopper/SaCCM_([1], [0], [1.0]).gif" width=200> <img src="docs/gif/walkerhopper/SaCCM_([2], [0], [1.0]).gif" width=200>
</p>

<p align=center>
<img src="docs/gif/walkerhopper/SaCCM([3], [0], [1.0]).gif" width=200> <img src="docs/gif/walkerhopper/SaCCM_([4], [0], [1.0]).gif" width=200> <img src="docs/gif/walkerhopper/SaCCM_([5], [0], [1.0]).gif" width=200> <img src="docs/gif/walkerhopper/SaCCM_([3, 4, 5], [3], [1.0]).gif" width=200>
</p>

* [ ] Multiple seeds training (5 seeds))
  Training task: [([0,1,2], [3], [1.0]),([1], [2], [1.0]),([0,1,2], [3], [1.0]), ([0],[2], [1.0])]
  Moderate task: [([0], [0], [1.0]),([1], [0], [1.0]), ([2], [0], [1.0])]
  Extreme task: [([3], [0], [1.0]), ([4], [0], [1.0]), ([5], [0], [1.0]), ([3,4,5], [3], [1.0])]
* [ ] Multiple seeds training (1 seeds))
  Training task: [([0,1,2], [3], [0.75,0.85,1.0,1.15,1.25]),([1], [2], [0.75,0.85,1.0,1.15,1.25]),([0,1,2], [3], [1.0]), ([0],[2], [0.75,0.85,1.0,1.15,1.25])]
  Moderate task: [([0], [0], [0.4, 0.5,1.5, 1.6]),([1], [0], [0.4, 0.5,1.5, 1.6]), ([2], [0], [0.4, 0.5,1.5, 1.6])]
  Extreme task: [([3], [0], [0.2,0.4,1.6,1.8]), ([4], [0], [0.2,0.4,1.6,1.8]), ([5], [0], [0.2,0.4,1.6,1.8]), ([3,4,5], [3], [0.2,0.4,1.6,1.8])]

#### Half-Cheetah ✔️

* [ ] Test SaCCM in task [([1.0], [1.0])], [([0.25], [0.25])], [([0.25], [1.75])], [([1.75], [0.25])], [([1.75], [1.75])], [([4.00], [1.00])]

#### Ant

* [ ] SAC single-task training setting: [([1.0],[1.0])], reward ctrl = 0.1, 0.05 (walking), 0.1 (rolling)

#### SlimHumanoid ✔️

During the training process, we alter the values of mass and damping. In the test setting, we test on previously unseen mass and damping values. In the SlimHumanoid environment, we found that in order to adapt to different mass and damping values, the TESAC/CCM/SaTESAC/SaCCM policy tends to learn only one skill, which is ***the Humanoid Robot crawling on the ground using one elbow**.* When the damping is relatively high (damping=1.6), the Humanoid Robot can crawl forward stably, but when the damping is low (damping=0.6), it tends to roll. The following shows the training and testing videos of SaCCM on different tasks (first 180 seconds, 6x speed):

<p align=center>
<img src="docs/gif/slimhumanoid/CCM_Moderate_Test_(mass=0.6,damping=0.6).gif" width=200> <img src="docs/gif/slimhumanoid/CCM_Moderate_Test_(mass=0.6,damping=1.5).gif" width=200> <img src="docs/gif/slimhumanoid/CCM_Moderate_Test_(mass=1.5,damping=0.6).gif" width=200> <img src="docs/gif/slimhumanoid/CCM_Moderate_Test_(mass=1.5,damping=1.5).gif" width=200> 
</p>

### Humanoid Standup ✔️

During the training process, we alter the values of mass and damping. In the test setting, we test on previously unseen mass and damping values. In the HumanoidStand environment, SaCCM and SaTESAC adapt better to varying mass and damping values compared to CCM and TESAC. From the example videos, the Humanoid Robot (using the CCM algorithm) takes more time struggling to stand up. This indicates that distinguishing different tasks according to the skills helps the RL agent to better execute the acquired skills and zero-shot generalise to different tasks. The following shows the training and testing videos of SaCCM on different tasks (first 90 seconds, 3x speed):

<p align=center>
<img src="docs/gif/humanoidstandup/SaCCM_Moderate_Test_(mass=0.6,damping=0.6).gif" width=200> <img src="docs/gif/humanoidstandup/SaCCM_Moderate_Test_(mass=0.6,damping=1.5).gif" width=200> <img src="docs/gif/humanoidstandup/SaCCM_Moderate_Test_(mass=1.5,damping=0.6).gif" width=200> <img src="docs/gif/humanoidstandup/SaCCM_Moderate_Test_(mass=1.5,damping=1.5).gif" width=200>
</p>

And some faliure cases of CCM:

<p align=center>
<img src="docs/gif/humanoidstandup/CCM_Moderate_Test_(mass=0.6,damping=0.6).gif" width=200> <img src="docs/gif/humanoidstandup/CCM_Moderate_Test_(mass=0.6,damping=1.5).gif" width=200> <img src="docs/gif/humanoidstandup/CCM_Moderate_Test_(mass=1.5,damping=0.6).gif" width=200> <img src="docs/gif/humanoidstandup/CCM_Moderate_Test_(mass=1.5,damping=1.5).gif" width=200>
</p>

### 📥 Requirements

We assume you have access to MuJoCo. We have shared some experiences on dealing with potential issues, see the file [install_mujoco-py.md](install_mujoco-py.md).

You may need to install some required dependencies when creating a conda environment:

```bash
python>=3.9

pip install torch>=1.13.0
pip install stable-baselines3[extra]
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install wandb
pip install seaborn
pip install tqdm
pip install psutil
pip install line_profiler # time cost test
pip install memory_profiler # memory cost test
pip install openpyxl
pip install imageio[ffmpeg]
pip install panda_gym
pip install gym==0.23.0
pip install beautifulsoup4 # for results visualization
```

We need to modify the code of two libraries (i.e., Gymnasium and Stable-Baselines3) in order to run SaMI. Please update them according to the code provided below:

```python
# stable_baselines3/common/vec_env/dummy_vec_env.py:70
# step_wait function
if self.buf_dones[env_idx]:
    # save final observation where user can get it, then reset
    self.buf_infos[env_idx]["terminal_observation"] = obs
    ans = self.envs[env_idx].reset()
    if type(ans) == tuple:
        obs, self.reset_infos[env_idx] = ans
    else:
        obs = ans
    # obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
self._save_obs(env_idx, obs)

# stable_baselines3/common/vec_env/dummy_vec_env.py:82
# reset function
for env_idx in range(self.num_envs):
    maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
    ans = self.envs[env_idx].reset(**maybe_options)
    if type(ans) == tuple:
        obs, info = ans
    else:
        obs = ans
    # self._save_obs(env_idx, obs)
    # obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(**maybe_options)
    self._save_obs(env_idx, obs)

# gymnasium/wrappers/time_limit.py:57
# step function
ans = self.env.step(action)
if len(ans) == 5:
    observation, reward, terminated,truncated, info = ans
else:
    observation, reward, terminated, info = ans
    truncated = False
self._elapsed_steps += 1

if self._elapsed_steps >= self._max_episode_steps:
    truncated = True
```

### 📄 Instructions

Download the code and produce the `output` folder, where all the outputs are going to be stored including train/eval logs.

```bash
git clone https://github.com/uoe-agents/SaMI.git
cd SaMI
mkdir output
```

You can run the code uing the configuration specified in `parsers.py` with:

```bash
python main.py
```

The `configs` folder contains bash scripts for all the algorithms used in the paper on the Panda-gym and Mujoco tasks as examples. You can run a specific configuration using the bash script, for example:

```bash
sh configs/mujoco_ant_train.sh
```

### 📈 Results and Plots From Paper

The data for the experiment results in the paper can be found here. These files contain the evaluation returns for all algorithms and seeds used to create Figures.

### 📎 Citation

```
@misc{yu2024skillaware,
      title={Skill-aware Mutual Information Optimisation for Generalisation in Reinforcement Learning}, 
      author={Xuehui Yu and Mhairi Dunion and Xin Li and Stefano V. Albrecht},
      year={2024},
      eprint={2406.04815},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
