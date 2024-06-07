
import os
import glob
import argparse
from datetime import datetime
import yaml
import sys
sys.path.append('/home/lixin/codes/rl_learning')
import tools

# # train
# test_env = {
#     'AntEnv':'[([0.75,0.85,1.0,1.15,1.25], [0.75,0.85,1.0,1.15,1.25])]',
#     'HalfCheetahEnv': '[([0.75,0.85,1.0,1.15,1.25],[0.75,0.85,1.0,1.15,1.25])]',
#     'CrippleAntEnv': '[([0, 1, 2], [0], [0.75,0.85,1.0,1.15,1.25])]',
#     'CrippleHalfCheetahEnv': '[([0, 1, 2, 3], [0], [0.40, 0.50])]',
#     'SlimHumanoidEnv': '[([0.8, 0.9, 1.0, 1.15, 1.25], [0.8, 0.9, 1.0, 1.15, 1.25])]',
#     'Hopper': '[([0.5, 0.75, 1.0, 1.25, 1.5], [0.5, 0.75, 1.0, 1.25, 1.5])]'
# }

# for env in test_env:
#     envs = eval(test_env[env])[0]
#     lens = [len(causal) for causal in envs]
#     count = 1
#     for _len in lens:  count *= _len
#     _test_envs = []
#     for i in range(count):
#         index = []
#         _i = i
#         for _len in lens:
#             index.append(_i % _len)
#             _i //= _len
        
#         cur = tuple()
#         for _i, _index in enumerate(index):
#             cur = cur + ([envs[_i][_index]],)
#         _test_envs.append(cur)
#     test_env[env] = str(_test_envs)

# test
# test_env = {
#     'AntEnv':'[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]',
#     'HalfCheetahEnv': '[([0.40, 0.50],[0.40, 0.50]),([0.40, 0.50],[1.50, 1.60]),([1.50, 1.60],[0.40, 0.50]),([1.50, 1.60],[1.50, 1.60])]',
#     'CrippleAntEnv': '[([3], [0],[0.4, 0.5]),([3], [0],[1.5, 1.6])]',
#     'CrippleHalfCheetahEnv': '[([4, 5], [0], [0.4, 0.5]),([4, 5], [0],[1.5, 1.6])]',
#     'SlimHumanoidEnv': '[([0.60, 0.70], [0.60, 0.70]),([0.60, 0.70], [1.50, 1.60]),([1.50, 1.60], [0.60, 0.70]),([1.50, 1.60], [1.50, 1.60])]',
#     'Hopper': '[([0.25, 0.375], [0.25, 0.375]),([0.25, 0.375], [1.75, 2.0]),([1.75, 2.0], [0.25, 0.375]),([1.75, 2.0], [1.75, 2.0])]'
# }

parser = argparse.ArgumentParser()
parser.add_argument('--date_start', default='2000-01-19-00:00:00-000000')
parser.add_argument('--date_end', default='2100-01-19-00:00:00-000000')
parser.add_argument('--cuda', default='0')
args = parser.parse_args()

files = glob.glob('output/2*')
start_date = datetime.strptime(args.date_start, '%Y-%m-%d-%H:%M:%S-%f')
end_date = datetime.strptime(args.date_end, '%Y-%m-%d-%H:%M:%S-%f')
for file in files:
    date = file.split('/')[-1]
    date = datetime.strptime(date, '%Y-%m-%d-%H:%M:%S-%f')
    if date >= start_date and date <= end_date:
        with open(os.path.join(file,'config.yaml'),'r',encoding='utf-8') as f:
            config = yaml.unsafe_load(f)
            env_name = config['model_parameters']['env_name']
            config['model_parameters']['train_envs'] = config['model_parameters']['test_envs']
            # if env_name in test_env:
            #     config['model_parameters']['test_envs'] = test_env[env_name]
        with open(os.path.join(file,'config.yaml'),'w',encoding='utf-8') as f:
            yaml.dump(config, f)
