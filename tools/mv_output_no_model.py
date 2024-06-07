#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   mv_output_no_model.py
@Author  :   lixin 
@Version :   1.0
@Desc    :   move no model output into a other path
'''
import os
import glob
import shutil

def mv_output_no_model(output_dir1, output_dir2):
    if not os.path.exists(output_dir1):
        os.mkdir(output_dir1)
    if not os.path.exists(output_dir2):
        os.mkdir(output_dir2)
    folders = glob.glob(os.path.join(output_dir1,"*"))
    for folder in folders:
        model_paths = os.path.join(folder, 'model*.zip')
        model_paths = glob.glob(model_paths)

        if len(model_paths) > 0: continue
        shutil.move(folder, folder.replace(output_dir1, output_dir2))

if __name__ == '__main__':
    mv_output_no_model('output','output_no_model')
