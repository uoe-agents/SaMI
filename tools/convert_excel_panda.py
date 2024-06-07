
import os
import sys
import glob
import argparse
import re
import yaml
from datetime import datetime
import pandas as pd
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)).replace('tools','')
sys.path.append(PROJECT_DIR)
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--date_start', default='2000-01-19-00:00:00-000000')
parser.add_argument('--date_end', default='2100-01-19-00:00:00-000000')
# parser.add_argument('--date_start', default='2024-05-13-18:54:44-095221')
# parser.add_argument('--date_end', default='2024-05-13-18:54:44-095221')

parser.add_argument('--output_dir', default='output')
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
        if not os.path.exists(os.path.join(file,'test_result.yaml')):
            continue
        with open(os.path.join(file,'test_result.yaml'),'r',encoding='utf-8') as f:
            test_result = yaml.safe_load(f)
        
        frictions = []
        masses = []
        for key in test_result:
            friction = re.search(r'friction:(.*?),mass:(.*?)$', key).group(1)
            mass = re.search(r'friction:(.*?),mass:(.*?)$', key).group(2)
            if float(friction) not in frictions:
                frictions.append(float(friction))
            if float(mass) not in masses:
                masses.append(float(mass))
            
        frictions.sort()
        masses.sort()
        
        push_table = {'fricion\\mass':frictions}
        for mass in masses:
            push_table[mass] = ['【空】'] * len(frictions)

        pick_table = {'fricion\\mass':frictions}
        for mass in masses:
            pick_table[mass] = ['【空】'] * len(frictions)

        roll_table = {'fricion\\mass':frictions}
        for mass in masses:
            roll_table[mass] = ['【空】'] * len(frictions)
        
        table = {'fricion\\mass':frictions}
        for mass in masses:
            table[mass] = ['【空】'] * len(frictions)

        for key in test_result:
            friction = float(re.search(r'friction:(.*?),mass:(.*?)$', key).group(1))
            mass = float(re.search(r'friction:(.*?),mass:(.*?)$', key).group(2))

            push_rate = test_result[key]['push_rate']
            roll_rate = test_result[key]['roll_rate']
            pick_and_place_rate = test_result[key]['pick_and_place_rate']
            success_rate = test_result[key]['success_rate']

            push_table[mass][frictions.index(friction)] = push_rate
            roll_table[mass][frictions.index(friction)] = roll_rate
            pick_table[mass][frictions.index(friction)] = pick_and_place_rate
            table[mass][frictions.index(friction)] = success_rate

        with pd.ExcelWriter(os.path.join(file,'test_result.xlsx')) as writer:
            df = pd.DataFrame(push_table)
            df.to_excel(writer, sheet_name='push_table', index=False)
            df = pd.DataFrame(roll_table)
            df.to_excel(writer, sheet_name='roll_table', index=False)
            df = pd.DataFrame(pick_table)
            df.to_excel(writer, sheet_name='pick_table', index=False)
            df = pd.DataFrame(table)
            df.to_excel(writer, sheet_name='success_rate', index=False)
        



        
