import pandas as pd

import os
from bs4 import BeautifulSoup




def extract_excel(html_file):

    output_dict = {}
    output = []    
    with open(html_file, 'r', encoding='utf-8') as f:
        data = f.read()
        soup = BeautifulSoup(data,'html.parser')
        table = soup.findAll('table')[0]
        if not os.path.exists('image'):
            os.mkdir('image')

        for i, child in enumerate(table.children):
            if child=='\n':
                continue
            if i == 1:
                continue
            
            child = list(child.children)
            child = list(filter(lambda x:x!='\n',child))
            env = child[0].get_text().strip()
            method = child[1].get_text().strip()

            train_reward = child[2].get_text().strip()
            test_reward = child[3].get_text().strip()
            output.append((env, method, train_reward, test_reward))
        
        output_dict['方法'] = list(set([d[1] for d in output]))
        output_dict['方法'].sort()
        for _data in output:
            env = _data[0]
            method = _data[1]
            if env+'_train' not in output_dict:
                output_dict[env+'_train'] = ['暂无'] * len(output_dict['方法'])
                output_dict[env+'_test'] = ['暂无'] * len(output_dict['方法'])
            try:
                output_dict[env+'_train'][output_dict['方法'].index(method)] = float(_data[2])
            except:
                output_dict[env+'_train'][output_dict['方法'].index(method)] = _data[2]
            try:
                output_dict[env+'_test'][output_dict['方法'].index(method)] = float(_data[3])
            except:
                output_dict[env+'_test'][output_dict['方法'].index(method)] = _data[3]
                
    
    df = pd.DataFrame(output_dict)
    df.to_excel(html_file.replace('.html','.xlsx'),index=False)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--html_file', default='output.html')
args = parser.parse_args()

ans = extract_excel(args.html_file)
# extract_image('/home/lixin/projects/rl_learning/outputskill_aware_buffer_size_1_seed_201_our_ourone_0509.html')