import os
from typing import Tuple
import numpy as np
import random
from torch.utils.data import Dataset
import pickle

from dataloader.result_loader import ResultFileLoader, PKLLoader
from visualization.utils import o3d_pcl, o3d_plot, pcl_filter, pcl_project
from nn.p4t.datasets.folder_list import *

class MMMosh(Dataset):
    def __init__(self, driver_path, clip_frames=5, train=True, **kwargs):
        self.driver_path = driver_path
        # range of frame index in a clip
        self.clip_step = kwargs.get('clip_step', 1)
        self.clip_range = clip_frames * self.clip_step
        self.train = train
        self.normal_scale = kwargs.get('normal_scale', 1)
        self.output_dim = kwargs.get('output_dim', 151)
        self.skip_head = kwargs.get('skip_head', 0)
        self.skip_tail = kwargs.get('skip_tail', 0)
        self.enable_sources = kwargs.get('enable_sources',["arbe","arbe_feature","master","sub2","kinect_color","kinect_pcl","kinect_pcl_remove_zeros","optitrack","mesh","mosh","mesh_param"])
        self.input_data = kwargs.get('input_data', 'mmWave')
        self.data_device = kwargs.get('data_device', 'arbe')
        self.num_points = 1024 if self.data_device=='arbe' else 4096
        self.test_data = kwargs.get('test_data', 'lab1')
        self.full_data = {}
        self.prep_pkl_data = kwargs.get('prep_data', True)
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0,]
        self.seq_loaders = []
        seq_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        # self.selected_dirs = ['2021-10-17_14-54-06_T']
        for d_path in map(str, self.driver_path.split(",")):
            if self.train:
                seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]
            else:
                seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]
        
        from multiprocessing.dummy import Pool
        pool = Pool(40)
        for path in seq_paths:
            pool.apply_async(self.creat_pkl, (path,))
        pool.close()
        pool.join()
        print('Done')

    def creat_pkl(self, path):
        import time
        seq_pkl = os.path.join(path, 'pkl_data/{}.pkl'.format(self.input_data))
        if os.path.exists(seq_pkl):
            os.remove(seq_pkl)
        try:
            # init result loader, reindex
            seq_loader = ResultFileLoader(root_path=path, skip_head=self.skip_head, enabled_sources=self.enable_sources)
            seq_len = len(seq_loader)
            if self.prep_pkl_data:
                self.full_data[path] = []
                for i in range(seq_len):
                    t_s = time.time()
                    self.full_data[path].append(self.load_data(seq_loader, i))
                    t_e = time.time()
                    print(path, i, t_e-t_s)
                if not os.path.exists(os.path.join(path, 'pkl_data')):
                    os.mkdir(os.path.join(path, 'pkl_data'))
                with open(seq_pkl, 'wb') as f:
                    pickle.dump(self.full_data[path], f)
        except Exception as e:
            print(e)
            # continue
            return
        if seq_len < 6:
            # continue
            return
        self.seq_loaders.append(seq_loader)
        self.index_map.append(self.index_map[-1] + seq_len)
        print(path, 'done')

    def pad_data(self, data):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        return data[r, :]

    def global_to_seq_index(self, global_idx: int) -> Tuple[int, int]:
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                # avoid out of range error
                return seq_idx, frame_idx if frame_idx >= self.clip_range else self.clip_range - 1
        raise IndexError

    def load_data(self, seq_loader, idx):
        if seq_loader.root_path in self.full_data.keys() and idx in self.full_data[seq_loader.root_path].keys():
            return self.full_data[seq_loader.root_path][idx]
        
        frame, info = seq_loader[idx]
        arbe_pcl = frame["arbe"]
        arbe_feature = frame["arbe_feature"][:, [0,4,5]]
        
        # normalization
        arbe_feature /= np.array([5e-38, 5, 150])

        # param: pose, shape
        if frame["mesh_param"] is None:
            return None, None
        
        opti_pcl = frame['optitrack']

        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]
        # mesh_vtx = frame["mesh_param"]["vertices"]
        mesh_jnt = frame["mesh_param"]["joints"]

        # 0 for female, 1 for male
        # gender = 1 if frame["information"].get("gender", "male") == "male" else 0

        # filter radar_pcl with optitrack bounding box
        arbe_data = pcl_filter(opti_pcl, np.hstack((arbe_pcl, arbe_feature)), 0.2)

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None

        bbox_center = ((mesh_jnt.max(axis=0) + mesh_jnt.min(axis=0))/2)[:3]
        arbe_data[:,:3] -= bbox_center
        mesh_pose[:3] -= bbox_center

        # padding
        arbe_data = self.pad_data(arbe_data)
        label = np.concatenate((mesh_pose / self.normal_scale, mesh_shape / self.normal_scale), axis=0)

        return arbe_data, label

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_loader = self.seq_loaders[seq_idx]
        clip = []

        data, label = self.load_data(seq_loader, frame_idx)

        while data is None:
            frame_idx = random.randint(self.clip_range-1, len(seq_loader)-1)
            data, label = self.load_data(seq_loader, frame_idx)

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.clip_step):
            # get xyz and features
            clip_data, _ = self.load_data(seq_loader, clip_id)
            # remove bad frame
            if clip_data is None:
                clip_data = data
            # padding
            clip.append(clip_data)
        clip.append(data)

        clip = np.asarray(clip, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)
        if True in np.isnan(label):
            label = np.nan_to_num(label)

        return clip, label, (seq_idx, frame_idx)


class MMFusion(MMMosh):
    def __init__(self, driver_path, clip_frames=5, train=True, **kwargs):
        enable_sources = ["arbe","master","kinect_color","mesh","mosh","mesh_param"]
        super().__init__(driver_path, clip_frames, train, enable_sources=enable_sources, **kwargs)

    def load_data(self, seq_loader, idx):
        # if seq_loader.root_path in self.full_data.keys() and idx in self.full_data[seq_loader.root_path].keys():
        if seq_loader.root_path in self.full_data.keys() and idx < len(self.full_data[seq_loader.root_path]):
            return self.full_data[seq_loader.root_path][idx]
        
        frame, info = seq_loader[idx]
        arbe_pcl = frame["arbe"]

        # param: pose, shape
        if frame["mesh_param"] is None:
            return None, None
        
        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]
        # mesh_vtx = frame["mesh_param"]["vertices"]
        mesh_joint = frame["mesh_param"]["joints"]
        rgb_data = frame['master_color']
        trans_mat = seq_loader.trans['kinect_master']

        # filter radar_pcl with optitrack bounding box
        arbe_data = pcl_filter(mesh_joint, arbe_pcl, 0.2)

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None
        
        mkv_fname = os.path.join(seq_loader.root_path, 'kinect/master/out.mkv')
        arbe_data = pcl_project(arbe_data, rgb_data, trans_mat, mkv_fname)

        bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0))/2)[:3]
        arbe_data[:,:3] -= bbox_center
        mesh_pose[:3] -= bbox_center

        # padding
        arbe_data = self.pad_data(arbe_data)
        label = np.concatenate((mesh_pose, mesh_shape), axis=0)

        return arbe_data, label


class MMMoshPKL(MMMosh):
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = []
        seq_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        for d_path in map(str, self.driver_path.split(",")):
            seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]

        # init result loader, reindex
        for path in seq_paths:
            pkl_fname = os.path.join(path, 'pkl_data/data.pkl')
            try:
                with open(pkl_fname, "rb") as f:
                    seq_data = pickle.load(f, encoding='bytes')
            except Exception as e:
                print(e)
                continue
            seq_len = len(seq_data) - self.skip_head
            if seq_len > 6:
                self.seq_paths.append(pkl_fname)
                self.index_map.append(self.index_map[-1] + seq_len)

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        frame_idx += self.skip_head
        seq_fname = self.seq_paths[seq_idx]
        with open(seq_fname, "rb") as f:
            seq = pickle.load(f, encoding='bytes')
        
        clip = []
        data = seq[frame_idx]

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.clip_step):
            # get xyz and features
            clip_data = seq[clip_id]
            # clip padding
            clip.append(clip_data[self.data_device])
        clip.append(data[self.data_device])
        clip = np.asarray(clip, dtype=np.float32)
        label = np.asarray(data['mesh_param'], dtype=np.float32)
        if True in np.isnan(clip):
            label = np.nan_to_num(clip)
        if True in np.isnan(label):
            label = np.nan_to_num(label)
            
        return clip, label, (seq_idx, frame_idx)


class MMFusionPKL(MMMoshPKL):
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = []
        seq_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        for d_path in map(str, self.driver_path.split(",")):
            seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]

        # init result loader, reindex
        for path in seq_paths:
            try:
                with open(os.path.join(path, 'pkl_data/arbe.pkl'), "rb") as f:
                    seq_data = pickle.load(f, encoding='bytes')
            except Exception as e:
                print(e)
                continue
            seq_len = len(seq_data) - self.skip_head
            if seq_len > 6:
                self.seq_paths.append(path)
                self.index_map.append(self.index_map[-1] + seq_len)

    def load_data(self, seq_path, idx):
        from calib.utils import to_radar_rectified_trans_mat

        with open(os.path.join(seq_path, 'pkl_data/arbe.pkl'), 'rb') as f:
            arbe_data = pickle.load(f)
        with open(os.path.join(seq_path, 'pkl_data/rgb.pkl'), 'rb') as f:
            rgb_data = pickle.load(f)
        with open(os.path.join(seq_path, 'pkl_data/mesh.pkl'), 'rb') as f:
            mesh_data = pickle.load(f)
        trans_mat = to_radar_rectified_trans_mat(seq_path)['kinect_master']

        arbe_frame = arbe_data[idx][:, :3]
        rgb_frame = rgb_data[idx]
        mesh_frame = mesh_data[idx]

        mesh_params = mesh_frame["params"]
        mesh_joints = mesh_frame["joints"]

        # filter radar_pcl with optitrack bounding box
        arbe_frame = pcl_filter(mesh_joints, arbe_frame, 0.2)

        # remove bad frame
        if arbe_frame.shape[0] == 0:
            return None, None
        
        arbe_frame = pcl_project(arbe_frame, rgb_frame, trans_mat, os.path.join(seq_path, 'kinect/master/out.mkv'))

        bbox_center = ((mesh_joints.max(axis=0) + mesh_joints.min(axis=0))/2)[:3]
        arbe_frame[:,:3] -= bbox_center
        mesh_params[:3] -= bbox_center

        # padding
        arbe_frame = self.pad_data(arbe_frame)

        return arbe_frame, mesh_params

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_sequence_index(idx)
        frame_idx += self.skip_head
        seq_path = self.seq_paths[seq_idx]
        data = self.load_data(seq_path, frame_idx)
        clip = []
        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.clip_step):
            # get xyz and features
            clip_data = self.load_data(seq_path, clip_id)
            # clip padding
            clip.append(clip_data[self.data_device])
        clip.append(data[self.data_device])
        clip = np.asarray(clip, dtype=np.float32)
        label = np.asarray(data['mesh_param'], dtype=np.float32)
        if True in np.isnan(clip):
            label = np.nan_to_num(clip)
        if True in np.isnan(label):
            label = np.nan_to_num(label)
            
        return clip, label, (seq_idx, frame_idx)