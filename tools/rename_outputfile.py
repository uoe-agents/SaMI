import os
import re


def replaceDirName(rootDir):
    dirs = os.listdir(rootDir)
    for dir in dirs:
        print('oldname is:' + dir)                # 输出老的名字
        if "04-07" in dir and ":" in dir:
            string = dir.split(':')
            temp = string[0] + '-' + string[1]  + '-' + string[2]   # 主要目的是在数字统一为3位，不够的前面补0
            print('new is:' + temp) 
            oldname = os.path.join(rootDir, dir)      # 老文件夹的名字
            newname = os.path.join(rootDir, temp)     # 新文件夹的名字
            os.rename(oldname, newname)
if __name__ == '__main__':
    rootdir = '/home/yxue/rl_learning/output'
    replaceDirName(rootdir)
