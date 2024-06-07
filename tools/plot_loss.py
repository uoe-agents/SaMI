import re
save_prefix = '/home/lixin/codes/rl_learning/output/2024-03-13-15:12:08-029221'
# total_timesteps = 1701930

with open(f'{save_prefix}/log.txt','r',encoding='utf-8') as f:
# with open('/home/llm_user/index/rl_learning/out/PandaPush_friction_1_2023-11-29-23:18:39.out','r',encoding='utf-8') as f:
    data = f.read()
    ep_rew_mean = re.findall(r'rollout/ep_rew_mean : ([-+]?[0-9]+(\.[0-9]+)?)\n',data)
    total_time_steps = re.findall(r'time/total_timesteps : ([-+]?[0-9]+(\.[0-9]+)?)\n',data)

    ep_rew_mean = [float(item[0].strip()) for item in ep_rew_mean]
    total_time_steps = [int(item[0].strip()) for item in total_time_steps]

import matplotlib.pyplot as plt
import numpy as np
titles = ['ep_rew_mean']
# titles = ['ep_len_mean','ep_rew_mean','success_rate','discriminator_loss','encoder_loss','actor_loss','critic_loss']
for title in titles:
    
    # if title in {'ep_len_mean','ep_rew_mean'}:
    #     continue

    if title == 'ep_rew_mean':
        xlabel = 'total_time_steps'
        t = total_time_steps
    else:
        xlabel = 'n_updates'
        # t = np.arange(0, len(actor_loss))

    ep_rew_mean = np.array(ep_rew_mean)

    fig, ax = plt.subplots()
    ax.plot(t, eval(title))

    ax.set(xlabel=xlabel, ylabel=f'{title}',
        title=f'{title}')

    ax.grid()

    fig.savefig(f"{save_prefix}/images/{title}.png")
    plt.show()