
import random
import os
import glob
import yaml
import csv
import itertools

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, 'output_no_model/*')
heads = ['train_env', 'method', 'rate']
test_fricitions = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
test_masses = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]

test_envs = list(itertools.product(test_fricitions, test_masses))
for _env_info in test_envs:
    heads.append(f'f{_env_info[0]},m{_env_info[1]}')

# os.path.dirname(os.path.abspath(__file__))
with open('merge.csv','w',encoding='utf-8')as f:
    writer = csv.writer(f)
    writer.writerow(heads)
    lines = []
    files = glob.glob(data_dir)
    for file in files:
        config_file = os.path.join(file, 'config.yaml')
        result_file = os.path.join(file, 'test_result.json')
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = yaml.safe_load(f)
        
        model_parameters = config_data['model_parameters']['method'] + '-' + config_data['model_parameters']['encoder_eps_type']
        line1 = [config_data['model_parameters']['train_envs'],model_parameters,'successful_rate']
        for k in  result_data: line1.append(result_data[k]['success_rate'])
        line2 = [config_data['model_parameters']['train_envs'],model_parameters,'pickandplace_rate']
        for k in  result_data: line2.append(result_data[k]['pick_and_place_rate'])
        line3 = [config_data['model_parameters']['train_envs'],model_parameters,'push_rate']
        for k in  result_data: line3.append(result_data[k]['push_rate'])
        line4 = [config_data['model_parameters']['train_envs'],model_parameters,'roll_rate']
        for k in  result_data: line4.append(result_data[k]['roll_rate'])
        lines.append((line1,line2,line3,line4))

        # writer.writerow(line1)
        # writer.writerow(line2)
        # writer.writerow(line3)
        # writer.writerow(line4)
    lines.sort(key=lambda x:x[0][0] + x[0][1])
    for _lines in lines:
        writer.writerows(list(_lines))
