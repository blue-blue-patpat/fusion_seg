import os
from typing import Tuple
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
from multiprocessing import Process

from dataloader.result_loader import ResultFileLoader, PKLLoader
from visualization.utils import o3d_pcl, o3d_plot, pcl_filter, get_pcl_feature
from nn.p4t.datasets.folder_list import *

class MMBody(Dataset):
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
        self.enable_sources = kwargs.get('enable_sources',["arbe","arbe_feature","master","kinect_color","mesh","mosh","mesh_param"])
        self.input_data = kwargs.get('input_data', 'mmWave')
        self.data_device = kwargs.get('data_device', 'arbe')
        self.num_points = 1024 if self.data_device=='arbe' else 4096
        self.test_data = kwargs.get('test_data', 'lab1')
        self.full_data = {}
        self.use_pkl = kwargs.get('use_pkl', True)
        self.feature_type = kwargs.get('feature_type', 'arbe')
        if self.feature_type == 'arbe' or self.feature_type == 'rgb' or self.feature_type == 'feature_map':
            self.features = 3
        elif self.feature_type == 'image_feature':
            self.features = 1000
        else:
            self.features = 0
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = []
        seq_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        for d_path in map(str, self.driver_path.split(",")):
            seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]
        
        def write_pkl():
            data = []
            for i in range(len(seq_loader)):
                data.append(self.load_data(seq_loader, i))
            if not os.path.exists(os.path.join(path, 'pkl_data')):
                os.mkdir(os.path.join(path, 'pkl_data'))
            with open(seq_pkl, 'wb') as f:
                pickle.dump(data, f)
            print(path, 'Done')

        process_dict = {}
        for path in seq_paths:
            seq_pkl = os.path.join(path, 'pkl_data/{}_{}.pkl'.format(self.input_data, self.feature_type))
            try:
                if os.path.exists(seq_pkl) and self.use_pkl:
                    with open(seq_pkl, 'rb') as f:
                        seq_loader = pickle.load(f)[self.skip_head:-self.skip_tail-1]
                else:
                    # init result loader, reindex
                    seq_loader = ResultFileLoader(root_path=path, skip_head=self.skip_head, skip_tail=self.skip_tail, enabled_sources=self.enable_sources)
                    if self.use_pkl:
                        p = Process(target=write_pkl)
                        p.start()
                        process_dict.update({path:p})
            except Exception as e:
                print(e)
                continue

            if len(seq_loader) < 6:
                continue
            self.full_data.update({path:seq_loader})
            self.seq_paths.append(path)
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
        for path, p in process_dict.items():
            p.join()
            with open(os.path.join(path, 'pkl_data/{}_{}.pkl'.format(self.input_data, self.feature_type)), 'rb') as f:
                self.full_data[path] = pickle.load(f)

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

    def __len__(self):
        return self.index_map[-1]

    def load_data(self, seq_loader, idx):
        if isinstance(seq_loader, list):
            return seq_loader[idx]

        frame, info = seq_loader[idx]
        arbe_pcl = frame["arbe"]

        # param: pose, shape
        if frame["mesh_param"] is None:
            return None, None
        
        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]
        mesh_joint = frame["mesh_param"]["joints"]

        if self.feature_type == 'arbe':
            arbe_feature = frame["arbe_feature"][:, [0,4,5]]
            arbe_feature /= np.array([5e-38, 5, 150])
            # filter radar_pcl with bounding box
            arbe_data = pcl_filter(mesh_joint, np.hstack((arbe_pcl, arbe_feature)), 0.2)
        
        else:
            rgb_data = frame['master_color']
            trans_mat = seq_loader.trans['kinect_master']
            mkv_fname = os.path.join(seq_loader.root_path, 'kinect/master/out.mkv')
            arbe_data = pcl_filter(mesh_joint, arbe_pcl, 0.2)
            # transform radar pcl coordinate to kinect master
            trans_pcl = (arbe_data - trans_mat['t']) @ trans_mat['R']
            trans_joint = (mesh_joint - trans_mat['t']) @ trans_mat['R']
            arbe_data = get_pcl_feature(trans_pcl, rgb_data, trans_joint, mkv_fname, self.feature_type, visual=False)

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None

        bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0))/2)[:3]
        arbe_data[:,:3] -= bbox_center
        mesh_pose[:3] -= bbox_center

        # padding
        arbe_data = self.pad_data(arbe_data)
        label = np.concatenate((mesh_pose / self.normal_scale, mesh_shape / self.normal_scale), axis=0)

        return arbe_data, label

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        clip = []
        seq_loader =self.full_data[seq_path]
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


class MMFusion(MMBody):
    def __init__(self, driver_path, clip_frames=5, train=True, **kwargs):
        enable_sources = ["arbe","master","kinect_color","mesh","mosh","mesh_param"]
        super().__init__(driver_path, clip_frames, train, enable_sources=enable_sources, **kwargs)

    def load_data(self, seq_loader, idx):
        if isinstance(seq_loader, list):
            return seq_loader[idx]
        
        frame, info = seq_loader[idx]
        arbe_pcl = frame["arbe"]

        # param: pose, shape
        if frame["mesh_param"] is None:
            return None, None
        
        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]
        mesh_joint = frame["mesh_param"]["joints"]
        rgb_data = frame['master_color']
        trans_mat = seq_loader.trans['kinect_master']

        # filter radar_pcl with bounding box
        arbe_data = pcl_filter(mesh_joint, arbe_pcl, 0.2)

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None
        
        mkv_fname = os.path.join(seq_loader.root_path, 'kinect/master/out.mkv')
        # transform radar pcl coordinate to kinect master
        trans_pcl = (arbe_data - trans_mat['t']) @ trans_mat['R']
        trans_joint = (mesh_joint - trans_mat['t']) @ trans_mat['R']
        arbe_data = get_pcl_feature(trans_pcl, rgb_data, trans_joint, mkv_fname, use_conv=True, use_feature_map=True, visual=False)

        bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0))/2)[:3]
        arbe_data[:,:3] -= bbox_center
        mesh_pose[:3] -= bbox_center

        # padding
        arbe_data = self.pad_data(arbe_data)
        label = np.concatenate((mesh_pose, mesh_shape), axis=0)

        return arbe_data, label


class MMMoshPKL(MMBody):
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

