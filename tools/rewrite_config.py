
import os
import glob
import argparse
from datetime import datetime
import yaml
import sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
import tools

# train
# test_env
# test_env = {
#     'AntEnv':'[([0.5], [1.0]),([0.75], [1.0]),([1.0], [1.0]),([1.25], [1.0]),([1.5], [1.0])]',
#     'HalfCheetahEnv': '[([0.5], [1.0]),([0.75], [1.0]),([1.0], [1.0]),([1.25], [1.0]),([1.5], [1.0])]',
#     'CrippleAntEnv': '[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0])]',
#     'CrippleHalfCheetahEnv': '[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]),([5], [0], [1.0])]',
#     'SlimHumanoidEnv': '[([0.5], [1.0]),([0.75], [1.0]),([1.0], [1.0]),([1.25], [1.0]),([1.5], [1.0])]',
#     'HopperEnv': '[([0.5], [1.0]),([0.75], [1.0]),([1.0], [1.0]),([1.25], [1.0]),([1.5], [1.0])]'
# }


# # extrame_env
# test_env = {
#     'AntEnv':'[([0.4, 0.45, 0.5, 0.55, 0.60], [1.0])]',
#     'HalfCheetahEnv': '[([0.2, 0.3, 1.7, 1.8], [0.2, 0.3, 1.7, 1.8])]',
#     'CrippleAntEnv': '[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0])]',
#     'CrippleHalfCheetahEnv': '[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]),([5], [0], [1.0])]',
#     'SlimHumanoidEnv': '[([0.4, 0.5, 1.7, 1.8], [0.4, 0.5, 1.7, 1.8])]',
#     'HopperEnv': '[([0.4, 0.5, 1.7, 1.8], [0.4, 0.5, 1.7, 1.8])]'
# }

# # extrame_env
# test_env = {
#     'AntEnv':'[([0.4, 0.45, 0.5, 0.55, 0.60], [1.0])]',
#     'HalfCheetahEnv': '[([0.2, 0.3, 1.7, 1.8], [0.2, 0.3, 1.7, 1.8])]',
#     'CrippleAntEnv': '[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0])]',
#     'CrippleHalfCheetahEnv': '[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]),([5], [0], [1.0])]',
#     'SlimHumanoidEnv': '[([0.4, 0.5, 1.7, 1.8], [0.4, 0.5, 1.7, 1.8])]',
#     'HopperEnv': '[([0.4, 0.5, 1.7, 1.8], [0.4, 0.5, 1.7, 1.8])]'
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
#     'AntEnv':'[([0.4], [0.4]),([0.40], [1.60]),([1.60], [0.40]),([1.60], [1.60])]',
#     'HalfCheetahEnv': '[([0.40],[0.40]),([0.40],[1.60]),([1.60],[0.40]),([1.60],[1.60])]',
#     'CrippleAntEnv': '[([3], [0],[0.4]),([3], [0],[1.6])]',
#     'CrippleHalfCheetahEnv': '[([4], [0], [0.4]),([5], [0],[1.6])]',
#     'SlimHumanoidEnv': '[([0.60], [0.60]),([0.60], [1.60]),([1.60], [0.60]),([1.60], [1.60])]',
#     'HopperEnv': '[([0.25], [0.25]),([0.25], [2.0]),([2.0], [0.25]),([2.0], [2.0])]'
# }


# test_env = {
#     'AntEnv':'[([0.4, 0.5], [0.4, 0.5]),([0.40, 0.50], [1.50, 1.60]),([1.50, 1.60], [0.40, 0.50]),([1.50, 1.60], [1.50, 1.60])]',
#     'HalfCheetahEnv': '[([0.40, 0.50],[0.40, 0.50]),([0.40, 0.50],[1.50, 1.60]),([1.50, 1.60],[0.40, 0.50]),([1.50, 1.60],[1.50, 1.60])]',
#     'CrippleAntEnv': '[([3], [0],[0.4, 0.5]),([3], [0],[1.5, 1.6])]',
#     'CrippleHalfCheetahEnv': '[([4, 5], [0], [0.4, 0.5]),([4, 5], [0],[1.5, 1.6])]',
#     'SlimHumanoidEnv': '[([0.60, 0.70], [0.60, 0.70]),([0.60, 0.70], [1.50, 1.60]),([1.50, 1.60], [0.60, 0.70]),([1.50, 1.60], [1.50, 1.60])]',
#     'HopperEnv': '[([0.25, 0.375], [0.25, 0.375]),([0.25, 0.375], [1.75, 2.0]),([1.75, 2.0], [0.25, 0.375]),([1.75, 2.0], [1.75, 2.0])]'
# }

# # extrame
# test_env = {
#     'AntEnv':'[([0.4],[1.0]),([0.45],[1.0]),([0.50],[1.0]),([0.55],[1.0]),([0.60],[1.0])]',
#     'HalfCheetahEnv': '[([0.20, 0.20],[0.20, 0.30]),([0.20, 0.30],[1.70, 1.80]),([1.70, 1.80],[0.20, 0.30]),([1.70, 1.80],[1.70, 1.80])]',
#     'CrippleAntEnv': '[([3], [0],[0.2, 0.3]),([3], [0],[1.7, 1.8])]',
#     'CrippleHalfCheetahEnv': '[([4, 5], [0], [0.2, 0.3]),([4, 5], [0],[1.7, 1.8])]',
#     'SlimHumanoidEnv': '[([0.40, 0.50], [0.40, 0.40]),([0.40, 0.40], [1.70, 1.80]),([1.70, 1.80], [0.40, 0.50]),([1.70, 1.80], [1.50, 1.60])]',
#     'HopperEnv': '[([0.25, 0.375], [0.25, 0.375]),([0.25, 0.375], [1.75, 2.0]),([1.75, 2.0], [0.25, 0.375]),([1.75, 2.0], [1.75, 2.0])]'
# }

test_env = {
    'PandaPush-v3':'[(0, 1), (0, 5), (1, 1),(1, 5)]'
}

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
    try: date = datetime.strptime(date, '%Y-%m-%d-%H:%M:%S-%f')
    except: date = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S-%f')
    if date >= start_date and date <= end_date:
        with open(os.path.join(file,'config.yaml'),'r',encoding='utf-8') as f:
            config = yaml.unsafe_load(f)
            env_name = config['model_parameters']['env_name']
            if env_name in test_env:
                config['model_parameters']['test_envs'] = test_env[env_name]
            # config['model_parameters']['test_eps_num_per_env'] = 1
        with open(os.path.join(file,'config.yaml'),'w',encoding='utf-8') as f:
            yaml.dump(config, f)
