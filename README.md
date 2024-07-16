# Skill-aware Mutural Information (SaMI)

This is the official implementation of Skill-aware Mutural Information (SaMI) from the paper Skill-aware Mutual Information Optimisation for Generalisation in Reinforcement Learning. on arxiv: https://arxiv.org/pdf/2406.04815

by [Xuehui Yu](https://github.com/yuxuehui), [Mhairi Dunion](https://github.com/mhairidunion), [Xin Li](https://github.com/loxs123), Stefano V. Albrecht

**Challenge Problem**
Meta-Reinforcement Learning (Meta-RL) agents can struggle to operate across tasks with varying environmental features that require different optimal skills (i.e., different modes of behaviors). An ideal RL algorithm should be able to learn a single policy to perform multiple tasks and generalise to new environments.

https://github.com/user-attachments/assets/2d644ccb-851c-4053-bfc4-0923f5c26080

SaMI is a plug-and-play module that can be integrated with any Meta-RL algorithm. In this repository, we provide implementations of two baselines: [CCM](https://cdn.aaai.org/ojs/16914/16914-13-20408-1-2-20210518.pdf) and [TESAC](https://arxiv.org/pdf/1910.10897). We have equipped these baselines with SaMI, resulting in SaSSM and SaTESAC.

**TODO**

- [ ] Change name "SaSAC" to "SaTESAC";
- [ ] Change name "DominoHook" to "MujocoHook"
- [ ] Change name "train_env" to "train_task"
- [ ] Change name "test_env" to "test_task"
- [ ] Main part needs videos which show SaCCM can Push, Pick&Place

### 📈 Results and Plots From Paper

The data for the experiment results in the paper can be found here. These files contain the evaluation returns for all algorithms and seeds used to create Figures.


<table>
    <tr>
        <td width="6%" height="100%">
            Friction=5.0
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
        <td width="6%" height="100%">
            Friction=1.0
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
</table>


#### Crippled Half-Cheetah

During the training process, we alter the values of mass and damping, and randomly cripple ***one of the front leg rotors*** (rotor 0/1/3). In the test setting, we test on previously unseen mass and damping values. More importantly, in the moderate test setting, we randomly cripple ***one of the back leg rotors***. In the extreme test setting, we increase the difficulty by randomly crippling ***two rotors from both the front and back legs***. The following shows the training and testing videos of SaCCM on different tasks (first 60 seconds, 2x speed):

<p align=center>
<img src="docs/gif/crippledhalfcheetah/([3], [0], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([0], [0], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([1], [0], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([2], [0], [1.0]).gif" width=200>
</p>
<p align=center>
<img src="docs/gif/crippledhalfcheetah/([0, 3], [1], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([1, 4], [1], [1.0]).gif" width=200> <img src="docs/gif/crippledhalfcheetah/([4, 5], [1], [1.0]).gif" width=200>
</p>

#### Crippled Ant

#### Half-Cheetah

#### Ant

#### SlimHumanoid

#### Walker (New)

#### Crippled Walker (New)

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
