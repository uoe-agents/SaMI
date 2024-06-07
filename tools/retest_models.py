
import os
import glob
import argparse
from datetime import datetime
import time
import subprocess
import psutil

def mem_allow_num():
    """
    return: int
    """
    # 获取内存信息
    mem = psutil.virtual_memory()
    # 将剩余内存转换为GB单位
    mem_gb = mem.available / (1024 * 1024 * 1024)
    # 打印剩余内存（以GB为单位）
    num = (mem_gb - 20) // 3
    return num if num>0 else 0

def gpu_allow_num():
    """
    {gpu_id: num}
    """
    # 运行nvidia-smi命令获取GPU信息
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    output_str = output.decode('utf-8')  # 将字节流转换为字符串
    # 将输出按行分割并提取剩余内存值
    gpu_memory = [int(x) for x in output_str.strip().split('\n')]
    gpus = {}
    # 打印每个GPU的剩余内存
    for i, mem in enumerate(gpu_memory):
        print("GPU {}: 剩余内存 {:.2f} MB".format(i, mem))
        gpus[i] = mem // 1.5
    return gpus


parser = argparse.ArgumentParser()
parser.add_argument('--date_start', default='2000-01-19-00:00:00-000000')
parser.add_argument('--date_end', default='2100-01-19-00:00:00-000000')
parser.add_argument('--cuda', default='0')
args = parser.parse_args()

files = glob.glob('output/2*')
start_date = datetime.strptime(args.date_start, '%Y-%m-%d-%H:%M:%S-%f')
end_date = datetime.strptime(args.date_end, '%Y-%m-%d-%H:%M:%S-%f')

need_to_run = []
for file in files:
    date = file.split('/')[-1]
    try: date = datetime.strptime(date, '%Y-%m-%d-%H:%M:%S-%f')
    except: date = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S-%f')
    if date >= start_date and date <= end_date:
        need_to_run.append(file)

need_to_run.sort()

while len(need_to_run) > 0:
    gpu_num = gpu_allow_num()
    mem_num = mem_allow_num()
    _gpu_num = len(gpu_num)
    left_run_num = 0
    for i in range(_gpu_num):
        left_run_num += gpu_num[i]
    left_run_num = min(left_run_num, mem_num)
    
    gpu_id = 0
    while left_run_num > 0 and len(need_to_run) > 0 and gpu_num[gpu_id] > 0:
        test_log_file = os.path.join(need_to_run[0], 'log-test4.txt')
        model_paths = os.path.join(need_to_run[0], 'model_*.zip')
        model_paths = glob.glob(model_paths)
        if len(model_paths) == 0:
            need_to_run = need_to_run[1:]
            continue
        # print(test_log_file)
        if os.path.exists(test_log_file):
            with open(test_log_file,'r',encoding='utf-8') as f:
                data = f.read()

                if data.strip() == '':
                    os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} nohup python main.py --config_path {need_to_run[0]} &')
                    gpu_id += 1
                    gpu_id %= _gpu_num
                    gpu_num[gpu_id] -= 1
                    need_to_run = need_to_run[1:]
                    left_run_num -= 1
                else:
                    need_to_run = need_to_run[1:]
        else:
            os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} nohup python main.py --config_path {need_to_run[0]} &')
            gpu_id += 1
            gpu_id %= _gpu_num
            gpu_num[gpu_id] -= 1
            need_to_run = need_to_run[1:]
            left_run_num -= 1
    time.sleep(100) #

