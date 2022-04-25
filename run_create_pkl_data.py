# import pickle
# from time import time
# import torch
# import numpy as np
# from nn.p4t.datasets.depth_mesh import DepthMesh3D
# from nn.p4t.datasets.mmmesh import MMMesh3D
# import os
# from multiprocessing.dummy import Pool
# # 该函数的作用是对网络的输入和输出进行预处理并储存。
# # 若choice为1，则提取点云信息，若choice为2则提取RGBD信息
# choice = 1
# data_path = '/home/nesc525/drivers/1'
# if choice == 2:
#     dataset = DepthMesh3D(
#             root_path=data_path,
#             frames_per_clip=5,
#             step_between_clips=1,
#             num_points=4096,
#             normal_scale=1,
#             skip_head=0,
#             output_dim=158,
#             train=True
#     )
#     device = 'sub1'
#     choice_dir =['2021-10-22_16-56-25_T']
# if choice == 1:
#     dataset = MMMesh3D(
#         root_path=data_path,
#         frames_per_clip=5,
#         step_between_clips=1,
#         num_points=1024,
#         normal_scale=1,
#         skip_head=0,
#         output_dim=158,
#         train=True
#     )
#     choice_dir = []
# print(len(dataset))
# files=os.listdir(data_path)


# def pre_porocess(dataset):
#     i = 0
#     train_data = []
#     label_data = []
#     t = time()
#     pre_ID = 0
#     for cilp, label, (ID,_,_) in dataset:
#         if pre_ID != ID:
#         #if i==1:
#             os.makedirs(os.path.join(data_path,choice_dir[pre_ID],'rgbd_data',device))
#             with open(os.path.join(data_path,choice_dir[pre_ID],'rgbd_data',device,'X_data.pkl'),'wb') as f:
#                 pickle.dump(train_data,f)
#             with open(os.path.join(data_path,choice_dir[pre_ID],'rgbd_data',device,'y_data.pkl'),'wb') as f:
#                 pickle.dump(label_data,f)
#             train_data = []
#             label_data = []
#         train_data.append(cilp)
#         label_data.append(label)
#         print(i, time()-t)
#         t = time()
#         i += 1
#         pre_ID = ID
#     os.makedirs(os.path.join(data_path,choice_dir[-1],'rgbd_data_test',device))
#     with open(os.path.join(data_path,choice_dir[-1],'rgbd_data_test',device,'X_data.pkl'),'wb') as f:
#         pickle.dump(train_data,f)
#     with open(os.path.join(data_path,choice_dir[-1],'rgbd_data_test',device,'y_data.pkl'),'wb') as f:
#         pickle.dump(label_data,f)

# pool = Pool(40)
# pool.apply_async(pre_porocess, (dataset,))

# pool.close()
# pool.join()

from multiprocessing.dummy import Pool
import os
import pickle
import time
from dataloader.result_loader import ResultFileLoader
from visualization.utils import pcl_filter

def single_pkl_gen(root_path, f_loader=None):
    enable_source = ["arbe", "arbe_feature", "master", "sub1", "sub2", "kinect_pcl", "kinect_pcl_remove_zeros", "optitrack", "mesh", "mosh", "mesh_param"]
    if f_loader is None:
        f_loader = ResultFileLoader(root_path, skip_head=0, skip_tail=0, enabled_sources=enable_source)
    save_path = os.path.join(f_loader.root_path, 'pkl')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if os.path.exists(os.path.join(save_path, 'data.pkl')):
    #     return
    # arbe = []
    # kinect = []
    # optitrack = []
    # mesh = []

    # for i in range(500):
    for i in range(len(f_loader)):
        t_s = time.time()
        frame, _ = f_loader[i]
        arbe_pcl = pcl_filter(frame['optitrack'], frame['arbe'], 0.2)
        master_pcl = pcl_filter(frame['optitrack'], frame['master_pcl'], 0.2)
        sub1_pcl = pcl_filter(frame['optitrack'], frame['sub1_pcl'], 0.2)
        sub2_pcl = pcl_filter(frame['optitrack'], frame['sub2_pcl'], 0.2)
        # arbe.append(arbe_pcl)
        # kinect.append(kinect_pcl)
        # optitrack.append(frame['optitrack'])
        # mesh.append(frame['mesh_obj'])
        frame.update(dict(arbe=arbe_pcl, master_pcl=master_pcl, sub1_pcl=sub1_pcl, sub2_pcl=sub2_pcl))

        with open(os.path.join(save_path, 'id={}.pkl'.format(i)), 'wb') as f:
            pickle.dump(frame, f)
        t_e = time.time()
        print(root_path, i, t_e-t_s)
    print(root_path, 'done')
    # res = dict(arbe=arbe, kinect=kinect, optitrack=optitrack, mesh=mesh)
    # with open(os.path.join(save_path, 'data.pkl'), 'wb') as f:
    #     pickle.dump(res, f)
    
# parent_path = '/home/nesc525/drivers/1'
# path_list = []
# for p in os.listdir(parent_path):
#     if os.path.exists(os.path.join(parent_path, p, 'mosh')):
#         # single_pkl_gen(os.path.join(parent_path, p))
#         path_list.append(os.path.join(parent_path, p))
# # path_list = ['/home/nesc525/drivers/1/2021-10-18_09-56-13_T']

# pool = Pool(40)
# for path in path_list:
#     pool.apply_async(single_pkl_gen, (path,))
# pool.close()
# pool.join()
# print('Done')



from multiprocessing.dummy import Pool
import os
import pickle
import time
import numpy as np
from dataloader.result_loader import ResultFileLoader
from visualization.utils import pcl_filter
from nn.p4t.datasets.folder_list import *

def pad_data(data, num_points):
    if data.shape[0] > num_points:
        r = np.random.choice(data.shape[0], size=num_points, replace=False)
    else:
        repeat, residue = num_points // data.shape[0], num_points % data.shape[0]
        r = np.random.choice(data.shape[0], size=residue, replace=False)
        r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
    return data[r, :]

def comp_pkl_gen(root_path, f_loader=None):
    enable_source = ["arbe", "arbe_feature", "master", "sub2", "kinect_color", "kinect_pcl", "optitrack", "mesh", "mosh", "mesh_param"]
    if f_loader is None:
        f_loader = ResultFileLoader(root_path, enabled_sources=enable_source)
    save_path = os.path.join(root_path, 'pkl_data')
    # save_path = os.path.join(root_path, 'pkl_data_new')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_list = []
    for i in range(len(f_loader)):
        t_s = time.time()
        frame, _ = f_loader[i]
        arbe_pcl = frame["arbe"]
        arbe_feature = frame["arbe_feature"][:, [0,4,5]]

        k_master_pcl = frame["master_pcl"]
        k_master_color = frame["master_color"]
        k_master_color = k_master_color.reshape(len(k_master_pcl), 3)

        k_sub2_pcl = frame["sub2_pcl"]
        k_sub2_color = frame["sub2_color"]
        k_sub2_color = k_sub2_color.reshape(len(k_sub2_pcl), 3)

        if frame["mesh_param"] is None:
            continue

        opti_pcl = frame['optitrack']

        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]
        mesh_joints = frame["mesh_param"]["joints"]

        # filter radar_pcl with optitrack bounding box
        arbe_data = pcl_filter(opti_pcl, np.hstack((arbe_pcl, arbe_feature)), 0.2)
        master_data = pcl_filter(opti_pcl, np.hstack((k_master_pcl, k_master_color)), 0.2, 0.21)
        sub2_data = pcl_filter(opti_pcl, np.hstack((k_sub2_pcl, k_sub2_color)), 0.2, 0.21)
        # arbe_data = pcl_filter(mesh_joints, np.hstack((arbe_pcl, arbe_feature)), 0.2)
        # master_data = pcl_filter(mesh_joints, np.hstack((k_master_pcl, k_master_color)), 0.2, 0.21)
        # sub2_data = pcl_filter(mesh_joints, np.hstack((k_sub2_pcl, k_sub2_color)), 0.2, 0.21)
        # remove bad frames
        if arbe_data.shape[0] == 0 or sub2_data.shape[0] == 0:
            continue

        # normalization
        bbox_center = ((opti_pcl.max(axis=0) + opti_pcl.min(axis=0))/2)[:3]
        # bbox_center = ((mesh_joints.max(axis=0) + mesh_joints.min(axis=0))/2)[:3]
        arbe_data[:,:3] -= bbox_center
        master_data[:,:3] -= bbox_center
        sub2_data[:,:3] -= bbox_center
        mesh_pose[:3] -= bbox_center
        arbe_data[:,3:] /= np.array([5e-38, 5, 150])
        master_data[:,3:] /= np.array([256, 256, 256])
        sub2_data[:,3:] /= np.array([256, 256, 256])

        # padding
        arbe_data = pad_data(arbe_data, 1024)
        master_data = pad_data(master_data, 4096)
        sub2_data = pad_data(sub2_data, 4096)

        mesh_param = np.concatenate((mesh_pose, mesh_shape), axis=0)
        
        data = dict(
            arbe=arbe_data,
            kinect_master=master_data,
            kinect_sub2=sub2_data,
            optitrack=opti_pcl,
            mesh_param=mesh_param
        )
        data_list.append(data)
        t_e = time.time()
        print(root_path, i, t_e-t_s)
    with open(os.path.join(save_path, 'data.pkl'), 'wb') as f:
        pickle.dump(data_list, f)
    print(root_path, 'done')

if __name__ == "__main__":
    driver_paths = ['/home/nesc525/drivers/1','/home/nesc525/drivers/2','/home/nesc525/drivers/3']
    path_list = []
    for d_p in driver_paths:
        for p in os.listdir(d_p):
            if p in SMOKE_DIRS:
            # if p[:10] == '2022-03-29':
                path_list.append(os.path.join(d_p, p))
    # path_list = ['/home/nesc525/drivers/1/2021-10-18_10-03-20_E']

    pool = Pool(40)
    for path in path_list:
        comp_pkl_gen(path)
        # pool.apply_async(comp_pkl_gen, (path,))
    pool.close()
    pool.join()
    print('Done')


