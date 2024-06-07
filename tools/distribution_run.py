## 分布式调试程序

import time
import requests
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

    return mem_gb-20 # 保留20G内存 

def gpu_allow_num():
    """
    {gpu_id: num}
    """
    # 运行nvidia-smi命令获取GPU信息
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    output_str = output.decode('utf-8')  # 将字节流转换为字符串
    # 将输出按行分割并提取剩余内存值
    gpu_memory = [int(x)/1024-2 for x in output_str.strip().split('\n')]
    return gpu_memory # 保留2G显存

api1 = 'http://host:5000/update_servers'
while True:
    ans = requests.post(api1, data={'name':'82', 'gpu_stat':str(gpu_allow_num()), 'mem_stat':mem_allow_num()})
    try: messages = eval(ans.text)
    except: print(ans.text)
    for msg in messages['message']:
        os.system(msg)
    time.sleep(300)
