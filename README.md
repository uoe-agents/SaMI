# SaMI
This is the official implementation of Skill-aware Mutural Information (SaMI) from the paper Skill-aware Mutual Information Optimisation for Generalisation in Reinforcement Learning.

### Environment

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

#### mujoco

[mujoco Install](install_mujoco-py.md)

#### Modify the code of some libraries

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

### Download the code and run it

```bash
git clone https://github.com/uoe-agents/SaMI.git
cd SaMI
mkdir output

# panda
CUDA_VISIBLE_DEVICES=0 python main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaCCM \
    --buffer_size 1000 \
    --train_freq 128 \
    --gradient_steps 16 \
    --learning_rate 1e-3 \
    --batch_size 256 \
    --contrast_batch_size 256 \
    --encoder_tau 0.05 \
    --seed 100 \
    --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
    --test_eps_num_per_env 50 \
    --use_wandb \
    --time_step 1_000_000 \
    --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5)]"

# domino [ant env]
CUDA_VISIBLE_DEVICES=0 python main.py \
    --env_name AntEnv \
    --env_hook DominoHook \
    --method SaCCM \
    --adversarial_loss_coef 0.1 \
    --use_weighted_info_nce \
    --seed 421 \
    --use_wandb \
    --test_envs "[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0.75,0.85], [0.75,0.85]),([0.75,0.85], [1.0,1.15,1.25]),([1.0,1.15,1.25], [0.75,0.85]),([1.0,1.15,1.25], [1.0,1.15,1.25])]"
```

