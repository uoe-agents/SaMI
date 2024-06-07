# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus']=False#显示负号
import os

def load_file(out_file_name):
    x_ep_train = []
    x_ep_adapt = []
    x_dim = []
    flag_count = 0
    f = open(out_file_name)               # 返回一个文件对象 
    line = f.readline()               # 调用文件的 readline()方法 
    while line: 
        if "Logging" in line:
            if flag_count == 0: # training
                flag_count = 1
            else: # testing
                flag_count = 2
                break
        if "success_rate" in line:
            x_ep_train.append(float(line.split('|')[-2].strip()))
        if "total_timesteps" in line:
            x_dim.append(int(line.split('|')[-2].strip()))
        line = f.readline()
    # read test
    if flag_count == 2:
        while line: 
            if "success_rate" in line:
                x_ep_adapt.append(float(line.split('|')[-2].strip()))
            if "total_timesteps" in line:
                x_dim.append(int(line.split('|')[-2].strip()))
            line = f.readline()
    
    for i in range(int(len(x_ep_train)), int(len(x_dim))):
        # print(x_dim[i], i, x_dim[int(len(x_ep_train))-1], len(x_ep_train))
        # raise
        x_dim[i] = x_dim[i] + x_dim[int(len(x_ep_train))-1]

    f.close()  
    # mean
    x = np.array(x_ep_train + x_ep_adapt)
    y = np.array(x_dim)
    # raise
    return x, y

def plot_function_multi(x, y, label_name = [], save_dir = "default"):
    f = plt.figure(figsize=(10,5))
    plt.grid(linestyle = "--") 
    ax = plt.gca()
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    color_list = ['lightcoral', 'gold', 'lightseagreen', 'yellowgreen', 'mediumpurple']
    for i in range(len(y)):
        plt.plot(y[i],x[i],color=color_list[i],label=label_name[i],linewidth=1.5)
    plt.ylabel("Success Rate",fontsize=13,fontweight='bold')
    plt.xlabel("Time steps",fontsize=13,fontweight='bold')
    plt.title(save_dir)
    plt.legend(loc='upper left')  # 一般plt.plot(label) 有label时一起使用
    
    plt.savefig(save_dir + ".png") #建议保存为svg格式，再用inkscape转为矢量图emf后插入word中

if __name__ == "__main__":
    # out_file_name_train = [
    #                         '/scratch/yxue/rl_learning/output/51182.out',
    #                         '/scratch/yxue/rl_learning/output/51409.out',
    #                         '/scratch/yxue/rl_learning/output/51410.out',
    #                         '/scratch/yxue/rl_learning/output/51411.out'
    # ]
    # "PandaPush-v3", "PandaPushDense-v3", "PandaPushJoints-v3", "PandaPushJointsDense-v3"], save_dir ="SAC_Dense_Sparse_EE_Joints"
    
    # out_file_name_train = [
    #                         '/scratch/yxue/rl_learning/output/50315.out',
    #                         '/scratch/yxue/rl_learning/output/51380.out',
    #                         '/scratch/yxue/rl_learning/output/51381.out',
    #                         '/scratch/yxue/rl_learning/output/51382.out'
    # ]
    # plot_function_multi(x_list, y_list, label_name = ["Mass 1", "Mass 100", "Mass 1000", "Mass 10000"], save_dir ="SAC_Sparse_EE_Big_Mass")

    # out_file_name_train = [
    #                         '/scratch/yxue/rl_learning/output/51409.out',
    #                         '/scratch/yxue/rl_learning/output/51434.out',
    #                         '/scratch/yxue/rl_learning/output/51435.out',
    #                         '/scratch/yxue/rl_learning/output/51436.out'
    # ]
    # plot_function_multi(x_list, y_list, label_name = ["Mass 1", "Mass 100", "Mass 1000", "Mass 10000"], save_dir ="SAC_PandaPushDense_Sparse_EE_Big_Mass")
    
    # out_file_name_train = [
    #                         '/scratch/yxue/rl_learning/output/51412.out',
    #                         '/scratch/yxue/rl_learning/output/51437.out',
    #                         '/scratch/yxue/rl_learning/output/51438.out',
    #                         '/scratch/yxue/rl_learning/output/51439.out'
    # ]
    # plot_function_multi(x_list, y_list, label_name = ["Mass 1", "Mass 100", "Mass 1000", "Mass 10000"], save_dir ="SAC_PandaPickAndPlace_Sparse_EE_Big_Mass")

    # out_file_name_train = [
    #                         '/scratch/yxue/rl_learning/output/51382.out',
    #                         '/scratch/yxue/rl_learning/output/51436.out',
    #                         '/scratch/yxue/rl_learning/output/51439.out',
    #                         # '/scratch/yxue/rl_learning/output/51439.out'
    # ]
    # plot_function_multi(x_list, y_list, label_name = ["PandaPush", "PandaPushDense", "PandaPickAndPlace"], save_dir ="SAC_EE_Big_Mass_10000_diff_tasks")
    name = 'gravity_2g'
    #out_file_name_train = [
    #                        f'PandaPickAndPlaceDense_{name}.out',
    #                        f'PandaPickAndPlace_{name}.out',
    #                        f'PandaPush_{name}.out',
    #                        f'PandaPushDense_{name}.out'
    #]
    out_file_name_train = [
                            f'PandaPickAndPlace_{name}.out',
                            f'PandaPush_{name}.out',
    ]
    #out_file_name_train = [
    #                        f'logs1011/PandaPickAndPlace_{name}.out'
    #]

    x_list = []
    y_list = []
    for file_index in range(len(out_file_name_train)):
        if "err" in out_file_name_train[file_index]:
            pass
        else:
            _name = out_file_name_train[file_index]
            x, y = load_file(out_file_name = _name)
            x_list.append(x)
            y_list.append(y)

    plot_function_multi(x_list, y_list, label_name = ["PandaPickAndPlace", "PandaPush"], save_dir =name)
    #plot_function_multi(x_list, y_list, label_name = ["PandaPickAndPlace"], save_dir =name)

