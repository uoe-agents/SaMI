html_doc = """
<!DOCTYPE html>
<html>
<head>
<style>
  table {
    border-collapse: collapse;
    width: 50%;
    margin: 0 auto;
  }

  table, th, td {
    border: 1px solid black;
  }

  th, td {
    padding: 10px;
    text-align: center;
  }

  td {
    max-width: 200px;
    word-wrap: break-word;
  }

  img {
    max-width: 150px;
    max-height: 150px;
  }
</style>
</head>
<body>

<table>
  <tr>
    <th>环境</th>
    <th>方法</th>
    <th>训练回报</th>
    <th>测试回报</th>
    <th>tsne</th>
    <th>路径</th>
  </tr>
</table>
</body>
</html>
"""
import base64
def tr_html(tr):
    """
    tr [env0, method1, train_return2, test_return3, reward_plot4,
        tsne5, encoder_loss6, actor_loss7, critic_loss8, 
        weight9, path10]
    """
    images = []
    for i in range(4, 5):
        if os.path.exists(tr[i]):
            with open(tr[i], "rb") as f:
                images.append(base64.b64encode(f.read()).decode())
        else:
            images.append(tr[i])
    return f"""
      <tr>
        <td>{tr[0]}</td>
        <td>{tr[1]}</td>
        <td>{tr[2]}</td>
        <td>{tr[3]}</td>
        <td><img src="data:image/png;base64,{images[0]}" alt="Placeholder Image"></td>
        <td>{tr[5]}</td>
      <tr>
    """

METHOD2PARAMETERS = {
    'TESAC':['buffer_size','learning_rate','train_freq','gradient_steps','seed'],
    'OURSACFastBase':['buffer_size','learning_rate','train_freq','gradient_steps',
                      'contrast_frame_stack','encoder_eps_type','use_weighted_info_nce',
                      'contrast_batch_size','encoder_tau','seed'],
    'OURSACFastAll':['buffer_size','learning_rate','train_freq','gradient_steps',
                      'contrast_frame_stack','encoder_eps_type','use_weighted_info_nce',
                      'contrast_batch_size','contrast_buffer_type','encoder_tau','seed'],
    'OURSACFastOne':['buffer_size','learning_rate','train_freq','gradient_steps',
                      'contrast_frame_stack','encoder_eps_type','use_weighted_info_nce',
                      'contrast_batch_size','contrast_buffer_type','encoder_tau','seed']
}

import os
import sys
import glob
import argparse
import re
import yaml
from datetime import datetime
from bs4 import BeautifulSoup
sys.path.append('/home/lixin/work/rl_learning/rl_learning')
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--date_start', default='2000-01-19-00:00:00-000000')
parser.add_argument('--date_end', default='2100-01-19-00:00:00-000000')
parser.add_argument('--output_dir', default='output')
parser.add_argument('--log_type', default='train') # or test continue train
parser.add_argument('--cuda', default='0')
args = parser.parse_args()

files = glob.glob(f'{args.output_dir}/2*')
start_date = datetime.strptime(args.date_start, '%Y-%m-%d-%H:%M:%S-%f')
end_date = datetime.strptime(args.date_end, '%Y-%m-%d-%H:%M:%S-%f')
data = []
for file in files:
    date = file.split('/')[-1]
    try: date = datetime.strptime(date, '%Y-%m-%d-%H:%M:%S-%f')
    except: date = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S-%f')
    if date >= start_date and date <= end_date:
        with open(os.path.join(file,'config.yaml'),'r',encoding='utf-8') as f:
            config = yaml.unsafe_load(f)
            env_name = config['model_parameters']['env_name']
            method = config['model_parameters']['method']
            method_name = method + ';'
            for param_key in METHOD2PARAMETERS[method]:
                if param_key not in config['model_parameters']:
                    continue
                method_name += param_key + ':' + str(config['model_parameters'][param_key]) + ";"


        train_log_file = os.path.join(file, 'log.txt')
        test_log_file = os.path.join(file, 'log-test2.txt')


        with open(train_log_file,'r',encoding='utf-8') as f:
            log = f.read()
            train_reward = re.findall('rollout/ep_rew_mean : (-?\d+\.?\d*)',log)
            if len(train_reward) == 0:
                train_reward.append('暂无')
            else:
                train_reward = train_reward[len(train_reward)-5:]
                train_reward = [float(_r) for _r in train_reward]
                train_reward = sum(train_reward) / len(train_reward)

        if os.path.exists(test_log_file):
            with open(test_log_file, 'r', encoding='utf-8') as f:
                log = f.read()
                _test_reward = re.findall('traj_len (-?\d+\.?\d*)\n',log)
                if len(_test_reward) == 0:
                    test_reward = '暂无'
                else:
                    _test_reward = [float(_r) for _r in _test_reward]
                    test_reward = sum(_test_reward) / len(_test_reward)
                    
        if not os.path.exists(test_log_file) or test_reward == '暂无':
            with open(train_log_file, 'r', encoding='utf-8') as f:
                log = f.read()
                _test_reward = re.findall('traj_len (-?\d+\.?\d*)\n',log)
                if len(_test_reward) == 0:
                    test_reward = '暂无'
                else:
                    _test_reward = [float(_r) for _r in _test_reward]
                    test_reward = sum(_test_reward) / len(_test_reward)

        reward_plot = os.path.join(file, 'images/ep_rew_mean.jpg')
        tsne = os.path.join(file, 'tsne.jpg')
        encoder_plot = os.path.join(file, 'images/encoder_loss.jpg')
        actor_plot = os.path.join(file, 'images/actor_loss.jpg')
        critic_plot = os.path.join(file, 'images/critic_loss.jpg')
        weight_plot = os.path.join(file, 'images/weight.jpg')
        data.append((env_name, method_name, train_reward, test_reward,
                      tsne,file))

data.sort(key=lambda x:x[1] + x[0])

soup = BeautifulSoup(html_doc, 'html.parser')
table = soup.find('table')

for item in data:
    tr = tr_html(item)
    table.append(BeautifulSoup(tr, 'html.parser'))

with open('result.html','w',encoding='utf-8') as f:
    f.write(soup.prettify())
