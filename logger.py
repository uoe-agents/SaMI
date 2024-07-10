#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   config.py
@Author  :   Lixin & Xuehui 
@Version :   1.0
@Desc    :   config file
'''

import os
import yaml
import datetime
import logging
import zipfile
import sys

# os.environ["WANDB_MODE"] = "offline" # wandb离线
# os.environ["WANDB_MODE"] = "dryrun"

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import seaborn as sns
import pandas as pd
import glob

from tools.time import check_timeout
from tools.video import VideoRecorder
from tools.utils import set_random_seed

project_dir = os.path.dirname(os.path.abspath(__file__))

def file2zip(zip_name, folder_path, save_path):
    def add_folder_to_zip(zip_file, folder_path, save_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                save_file_path = os.path.join(save_path, os.path.relpath(file_path, folder_path))
                if file_path.split('.')[-1] != 'py': continue
                zip_file.write(file_path, save_file_path)

    with zipfile.ZipFile(zip_name, "w") as zip_file:
        add_folder_to_zip(zip_file, folder_path, save_path)

def moving_average(array, window_size):
    # 在数组两侧填充0，使其与原数组形状一致
    padding = window_size // 2
    padded_array = np.pad(array, padding, mode='edge')
    
    # 使用滑动窗口视图获取滑动窗口，并计算平均值
    windowed_array = sliding_window_view(padded_array, window_size)
    average = np.mean(windowed_array, axis=1)
    
    return average

class Manager:
    def __init__(self, args, save_path) -> None:
        self.model_parameters = args.__dict__
        self.save_path = save_path
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S-%f")
        self.sub_save_path = os.path.join(save_path, cur_time)
        self.save_model_path = os.path.join(self.sub_save_path, 'model.zip')
        self.log_path = os.path.join(self.sub_save_path, 'log.txt')
        self.code_path = os.path.join(self.sub_save_path, 'code.zip')
        self.config_path = os.path.join(self.sub_save_path, 'config.yaml')

        self.test_path = os.path.join(self.sub_save_path, 'test_result.yaml')
        self.test_tsne_path = os.path.join(self.sub_save_path, 'tsne.jpg')
        self.test_table_path = os.path.join(self.sub_save_path, 'tables')
        self.test_image_path = os.path.join(self.sub_save_path, 'images')
        self.test_video_path = os.path.join(self.sub_save_path, 'videos')
        print(self.__dict__)

    def setup(self):
        """
        setup log path, code path, model path
        """
        # sub model path
        if not os.path.exists(self.sub_save_path):
            os.mkdir(self.sub_save_path)
        
        # code path
        if not os.path.exists(self.code_path):
            folder_path = project_dir
            file2zip(self.code_path,folder_path,'./rl_learning') # 保存当前时刻代码文件

        # video path
        if not os.path.exists(self.test_video_path):
            os.mkdir(self.test_video_path)
        self.recoder = VideoRecorder(self.test_video_path)
        if not self.model_parameters['save_video']:
            self.recoder.disable()

        # config path
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                yaml.dump(self.__dict__, f)

        # log path
        if 'test' in self.log_path:
            self.log_file = open(self.log_path, 'w', encoding='utf-8')
        else:
            self.log_file = open(self.log_path, 'a', encoding='utf-8')

        sys.stdout = self.log_file

        # wandb
        if self.model_parameters['use_wandb']:
            import wandb
            self.wandb = wandb
            check_timeout(self.wandb.init, dict(
                # set the wandb project where this run will be logged
                project=self.model_parameters['wandb_project_name'],
                entity=self.model_parameters['wandb_team_name'],
                name=self._wandb_name(),
                # track hyperparameters and run metadata
                config=self.model_parameters
            ),20)
            # 设置超时时间为5秒
            self.wandb.config.log_timeout = 10
        
        # seed
        set_random_seed(self.model_parameters['seed'])

        # reset every rate
        self.success_rate = []
        self.actor_loss = []
        self.critic_loss = []
        self.encoder_loss = []
        self.ent_coef_loss = []
        self.ep_rew_mean = []
        self.weight = []

    def load_config(self, sub_save_path):
        """
        only load save_model_path and model_parameters
        """
        print('loading from config, origin model_parameters setting is out of date.')
        with open(os.path.join(sub_save_path, 'config.yaml'),'r',encoding='utf-8') as f:
            data = yaml.unsafe_load(f)
        ignore_keys = {'send_email_to_my_sister','target_email','config_path',
                       'use_continue_train','use_wandb','wandb_project_name',
                       'wandb_team_name','save_video','time_step'}
        for key,value in data['model_parameters'].items():
            if key in ignore_keys: continue
            self.model_parameters[key] = value

        self.sub_save_path = sub_save_path
        self.log_path = os.path.join(self.sub_save_path, 'log-test2.txt')
        self.code_path = os.path.join(self.sub_save_path, 'code.zip')
        self.config_path = os.path.join(self.sub_save_path, 'config.yaml')
        model_paths = os.path.join(self.sub_save_path, 'model_*.zip')
        model_paths = glob.glob(model_paths)
        model_paths = list(filter(lambda x:'continue' not in x, model_paths))    
        model_paths = [path.split('/')[-1] for path in model_paths]

        model_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else -1)
        if len(model_paths) > 0: self.save_model_path = os.path.join(self.sub_save_path, model_paths[len(model_paths)//2])
        else: self.save_model_path = os.path.join(self.sub_save_path, 'model.zip')
        self.test_path = os.path.join(self.sub_save_path, 'test_result.yaml')
        self.test_tsne_path = os.path.join(self.sub_save_path, 'tsne.jpg')
        self.test_table_path = os.path.join(self.sub_save_path, 'tables')
        self.test_image_path = os.path.join(self.sub_save_path, 'images')
        self.test_video_path = os.path.join(self.sub_save_path, 'videos')

    def setup_logger(self):
        logger = logging.getLogger()
        name_to_value, name_to_count = {},{}
        def info(text,exclude=None):
            print(text)

        def record(key, value, exclude=None):
            name_to_value[key] = value

        def record_mean(key, value, exclude=None):
            if value is None:
                name_to_value[key] = None
                return
            old_val, count = name_to_value.get(key, 0), name_to_count.get(key, 0)
            name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
            name_to_count[key] = count + 1

        def dump(step):
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S-%f")
            logger.info('#'*10 + 'step:' + '%07d'%(step) + '#'*10 )
            logger.info('cur_time : ' + cur_time)
            for key in name_to_value:
                logger.info(str(key) + ' : ' + str(name_to_value[key]))
            logger.info('#'*33)
            if self.model_parameters['use_wandb']:
                need_to_plot_keys = {'rollout/success_rate','rollout/ep_rew_mean','train/actor_loss',
                                    'train/critic_loss','train/ent_coef_loss',
                                    'train/encoder_loss','train/weight'}
                need_to_plot = {k:v for k,v in name_to_value.items() if k in need_to_plot_keys}
                check_timeout(self.wandb.log, dict(data=need_to_plot,step=step),2)

            self.plot_loss(name_to_value, step)

            name_to_value.clear()
            sys.stdout.flush()
        
        setattr(logger, 'record', record)
        setattr(logger, 'record_mean', record_mean)
        setattr(logger, 'dump', dump)
        setattr(logger, 'info', info)

        return logger

    def record_video(self, env):
        self.recoder.record(env)
    
    def save_video(self, file_name):
        self.recoder.save(file_name)
    
    def reset_video(self):
        self.recoder.reset()

    def disable_video(self):
        self.recoder.disable()

    def enable_video(self):
        self.recoder.enable()

    def plot_loss(self, name_to_value, step):
        if 'rollout/success_rate' in name_to_value:
            self.success_rate.append((step,name_to_value['rollout/success_rate']))
        if 'train/actor_loss' in name_to_value:
            self.actor_loss.append((step,name_to_value['train/actor_loss']))
        if 'train/critic_loss' in name_to_value:
            self.critic_loss.append((step,name_to_value['train/critic_loss']))
        if 'train/ent_coef_loss' in name_to_value:
            self.ent_coef_loss.append((step,name_to_value['train/ent_coef_loss']))
        if 'train/encoder_loss' in name_to_value:
            self.encoder_loss.append((step,name_to_value['train/encoder_loss']))
        if 'rollout/ep_rew_mean' in name_to_value:
            self.ep_rew_mean.append((step,name_to_value['rollout/ep_rew_mean']))
        if 'train/weight' in name_to_value:
            self.weight.append((step,name_to_value['train/weight']))

        if not os.path.exists(self.test_image_path):
            os.mkdir(self.test_image_path)
        
        if len(self.success_rate) > 0:
            path = os.path.join(self.test_image_path, 'success_rate.jpg')
            self._plot_loss(self.success_rate, path, 'time_steps', 'success_rate')
        if len(self.actor_loss) > 0:
            path = os.path.join(self.test_image_path, 'actor_loss.jpg')
            self._plot_loss(self.actor_loss, path, 'time_steps', 'actor_loss')
        if len(self.critic_loss) > 0:
            path = os.path.join(self.test_image_path, 'critic_loss.jpg')
            self._plot_loss(self.critic_loss, path, 'time_steps', 'critic_loss')
        if len(self.ent_coef_loss) > 0:
            path = os.path.join(self.test_image_path, 'ent_coef_loss.jpg')
            self._plot_loss(self.ent_coef_loss, path, 'time_steps', 'ent_coef_loss')
        if len(self.encoder_loss) > 0:
            path = os.path.join(self.test_image_path, 'encoder_loss.jpg')
            self._plot_loss(self.encoder_loss, path, 'time_steps', 'encoder_loss')
        if len(self.ep_rew_mean) > 0:
            path = os.path.join(self.test_image_path, 'ep_rew_mean.jpg')
            self._plot_loss(self.ep_rew_mean, path, 'time_steps', 'ep_rew_mean')
        if len(self.weight) > 0:
            path = os.path.join(self.test_image_path, 'weight.jpg')
            self._plot_loss(self.weight, path, 'time_steps', 'weight')
            
    def _plot_loss(self,data, path, x_name,y_name, window=19):
        if len(data) <= window: return
        x = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        y = moving_average(y, window)
        plt.plot(x, y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.savefig(path)
        plt.clf()

    def close(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
        sys.stdout = sys.__stdout__
        if self.model_parameters['use_wandb']:
            self.wandb.finish()
    
    def plot_scatter(self,X,y,class_names, alpha):

        # 数据准备
        # 假设你有一个特征矩阵X，它包含你要进行降维的数据
        # 你还有一个目标数组y，它包含数据点的类标签
        # alpha[:] = 1
        # 创建t-SNE模型
        tsne = TSNE(n_components=2, random_state=42)
        # 使用fit_transform方法将特征矩阵X转换为二维
        X_embedded = tsne.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca_embedded = pca.fit_transform(X)

        # 可视化散点图

        # 创建一个颜色列表，用于给不同类别的点设置不同的颜色
        color_list = plt.cm.Set1(np.linspace(0, 1, len(class_names)))

        # plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            # 选择属于特定类别的数据点
            class_X = X_embedded[y == i]
            alpha_X = alpha[y==i]
            # 绘制散点图，并设置点的标签
            plt.scatter(class_X[:, 0], class_X[:, 1], color=color_list[i], label=class_name, s=3, alpha=alpha_X)

        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_list[i], markersize=8) for i in range(len(class_names))], labels=class_names)
        if self.model_parameters['use_wandb']:
            # 将图表记录到W&B
            check_timeout(self.wandb.log, dict(data={"tsne": self.wandb.Image(plt)}))
            # 关闭图表
        plt.savefig(self.test_tsne_path, dpi=300)
        plt.clf()

        # plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            # 选择属于特定类别的数据点
            class_X = X_pca_embedded[y == i]
            alpha_X = alpha[y==i]
            # 绘制散点图，并设置点的标签
            plt.scatter(class_X[:, 0], class_X[:, 1], color=color_list[i], label=class_name, s=3, alpha=alpha_X)

        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_list[i], markersize=8) for i in range(len(class_names))], labels=class_names)
        if self.model_parameters['use_wandb']:
            # 将图表记录到W&B
            check_timeout(self.wandb.log, dict(data={"pca": self.wandb.Image(plt)}))
            # 关闭图表
        plt.savefig(self.test_tsne_path.replace('tsne','pca'), dpi=300)
        plt.clf()


    def plot_table(self,data,row_names,col_names,table_name):
        if not os.path.exists(self.test_table_path):
            os.mkdir(self.test_table_path)
        df = pd.DataFrame(data, index=row_names, columns=col_names)
        df.to_excel(os.path.join(self.test_table_path,table_name+'.xlsx'))
        # if self.model_parameters['use_wandb']:
        #     new_df = {'':row_names}
        #     for col_i,col_name in enumerate(col_names):
        #         new_df[col_name] = data[:,col_i]
        #     df = pd.DataFrame(new_df)
        #     tbl = self.wandb.Table(dataframe = df)
        #     check_timeout(self.wandb.log, dict(data = {table_name:tbl}))

    def _wandb_name(self):
        name = self.model_parameters['method']
        return name