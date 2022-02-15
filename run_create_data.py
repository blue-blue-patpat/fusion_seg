import pickle
from time import time
import torch
import numpy as np
from nn.p4t.datasets.depth_mesh import DepthMesh3D
from nn.p4t.datasets.mmmesh import MMMesh3D
import os
from multiprocessing.dummy import Pool
# 该函数的作用是对网络的输入和输出进行预处理并储存。
# 若choice为1，则提取点云信息，若choice为2则提取RGBD信息
choice = 1
data_path = '/home/nesc525/drivers/1'
if choice == 2:
    dataset = DepthMesh3D(
            root_path=data_path,
            frames_per_clip=5,
            step_between_clips=1,
            num_points=4096,
            normal_scale=1,
            skip_head=0,
            output_dim=158,
            train=True
    )
    device = 'sub1'
    choice_dir =['2021-10-22_16-56-25_T']
if choice == 1:
    dataset = MMMesh3D(
        root_path=data_path,
        frames_per_clip=5,
        step_between_clips=1,
        num_points=1024,
        normal_scale=1,
        skip_head=0,
        output_dim=158,
        train=True
    )
    choice_dir = []
print(len(dataset))
files=os.listdir(data_path)


def pre_porocess(dataset):
    i = 0
    train_data = []
    label_data = []
    t = time()
    pre_ID = 0
    for cilp, label, (ID,_,_) in dataset:
        if pre_ID != ID:
        #if i==1:
            os.makedirs(os.path.join(data_path,choice_dir[pre_ID],'rgbd_data',device))
            with open(os.path.join(data_path,choice_dir[pre_ID],'rgbd_data',device,'X_data.pkl'),'wb') as f:
                pickle.dump(train_data,f)
            with open(os.path.join(data_path,choice_dir[pre_ID],'rgbd_data',device,'y_data.pkl'),'wb') as f:
                pickle.dump(label_data,f)
            train_data = []
            label_data = []
        train_data.append(cilp)
        label_data.append(label)
        print(i, time()-t)
        t = time()
        i += 1
        pre_ID = ID
    os.makedirs(os.path.join(data_path,choice_dir[-1],'rgbd_data_test',device))
    with open(os.path.join(data_path,choice_dir[-1],'rgbd_data_test',device,'X_data.pkl'),'wb') as f:
        pickle.dump(train_data,f)
    with open(os.path.join(data_path,choice_dir[-1],'rgbd_data_test',device,'y_data.pkl'),'wb') as f:
        pickle.dump(label_data,f)

pool = Pool(40)
pool.apply_async(pre_porocess, (dataset,))

pool.close()
pool.join()
