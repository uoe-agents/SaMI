import zipfile
import os
import sys
import argparse
from datetime import datetime

def file2zip(zip_name, folder_path, save_path,files_to_exclude=None, date_start = None, date_end = None):
    def add_folder_to_zip(zip_file, folder_path, save_path):
        for root, dirs, files in os.walk(folder_path):
            path = root.split('/')
            if len(path) >= 2:
                cur_date = datetime.strptime(path[1], '%Y-%m-%d-%H:%M:%S-%f')
                if cur_date < date_start or cur_date > date_end:
                    continue
            # print(path)
            if len(root.split('/')) == 2 and 'test_result.xlsx' not in files:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                save_file_path = os.path.join(save_path, os.path.relpath(file_path, folder_path))
                if file_path.split('.')[-1] == 'zip': continue
                if file_path.split('.')[-1] == 'txt': continue
                zip_file.write(file_path, save_file_path)

    with zipfile.ZipFile(zip_name, "w") as zip_file:
        add_folder_to_zip(zip_file, folder_path, save_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='output')
    parser.add_argument('--date_start', default='2024-02-29-03:55:50-184422')
    parser.add_argument('--date_end', default='2025-02-28-03:55:50-184423')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    input_path = args.input_path
    output_path = input_path + '.zip'
    if sys.platform.startswith('win'):
        out_dir = input_path.split('\\')[-1]
    else: # maybe linux
        out_dir = input_path.split('/')[-1]
    date_start = datetime.strptime(args.date_start, '%Y-%m-%d-%H:%M:%S-%f')
    date_end = datetime.strptime(args.date_end, '%Y-%m-%d-%H:%M:%S-%f')
    file2zip(output_path, input_path, out_dir, date_start=date_start, date_end=date_end)
