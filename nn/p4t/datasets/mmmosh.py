import os
from typing import Tuple
import numpy as np
import random
from sympy import sequence
from torch.utils.data import Dataset
import pickle

from dataloader.result_loader import ResultFileLoader, PKLLoader
from visualization.utils import pcl_filter
from nn.p4t.datasets.folder_list import *

class MMMosh(Dataset):
    def __init__(self, driver_path, frames_per_clip=5, step_between_clips=1, num_points=1024,
            train=True, normal_scale = 1, output_dim=151, skip_head=0, skip_tail=0, data_type='arbe'):
        self.driver_path = driver_path
        # range of frame index in a clip
        self.step_between_clips = step_between_clips
        self.clip_range = frames_per_clip * step_between_clips
        self.num_points = num_points
        self.train = train
        self.normal_scale = normal_scale
        self.output_dim = output_dim
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0,]
        self.sequence_loaders = []
        sequences_paths = []
        self.sequence_path = []
        self.selected_dirs = TRAIN_DIRS if self.train else LAB1_DIRS
        self.selected_dirs = RAIN_DIRS
        # self.selected_dirs = ['2021-10-17_14-54-06_T']
        for d_path in map(str, self.driver_path.split(",")):
            if self.train:
                sequences_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]
            else:
                sequences_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]

        for idx, path in enumerate(sequences_paths):
            # init result loader, reindex
            try:
                sequence_loader = ResultFileLoader(root_path=path, skip_head=self.skip_head, enabled_sources=["arbe", "arbe_feature", "master", "sub2", "kinect_pcl", "kinect_pcl_remove_zeros", "optitrack", "mesh", "mosh", "mesh_param"])
            except Exception as e:
                print(e)
                continue
            if len(sequence_loader) < 6:
                continue
            sequence_loader.skip_head = 0
            self.sequence_loaders.append(sequence_loader)
            self.sequence_path.append(path)
            sequence_len = len(sequence_loader)
            # if int(path[32:34]) >= 22:
            #     sequence_len = min(sequence_len, 300)
            self.index_map.append(self.index_map[-1] + sequence_len)

    def pad_data(self, data):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        return data[r, :]

    def global_to_sequence_index(self, global_idx: int) -> Tuple[int, int]:
        for sequence_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[sequence_idx], self.index_map[sequence_idx+1]):
                frame_idx = global_idx - self.index_map[sequence_idx]
                # avoid out of range error
                return sequence_idx, frame_idx if frame_idx >= self.clip_range else self.clip_range - 1
        raise IndexError

    def load_data(self, sequence_loader, idx):
        frame, info = sequence_loader[idx]
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
        sequence_idx, frame_idx = self.global_to_sequence_index(idx)
        sequence_loader = self.sequence_loaders[sequence_idx]
        clip = []

        data, label = self.load_data(sequence_loader, frame_idx)

        while data is None:
            frame_idx = random.randint(self.clip_range-1, len(sequence_loader)-1)
            data, label = self.load_data(sequence_loader, frame_idx)

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.step_between_clips):
            # get xyz and features
            clip_data, clip_label = self.load_data(sequence_loader, clip_id)
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

        return clip, label, (sequence_idx, frame_idx)


class MMMoshPKL(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=1024,
            train=True, normal_scale=1, output_dim=151, skip_head=0, skip_tail=0):
        super(MMMoshPKL, self).__init__()
        self.driver_path = root_path
        # range of frame index in a clip
        self.step_between_clips = step_between_clips
        self.clip_range = frames_per_clip * step_between_clips
        self.num_points = num_points
        self.train = train
        self.normal_scale = normal_scale
        self.output_dim = output_dim
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0,]
        sequence_list = []
        self.sequence_list = []
        self.selected_dirs = ['2021-10-17_14-54-06_T','2021-10-18_09-50-33_T','2021-10-18_09-48-31_T']
        # self.selected_dirs = [p for p in os.listdir('/home/nesc525/drivers/2') if p[:10] == '2022-03-25']
        for root_path in map(str, self.driver_path.split(",")):
            if self.train:
                sequence_list += [os.path.join(root_path, p) for p in os.listdir(root_path) if p in self.selected_dirs]
            else:
                sequence_list += [os.path.join(root_path, p) for p in os.listdir(root_path) if p in self.selected_dirs]

        for p in sequence_list:
            pkl_path = os.path.join(p, 'pkl')
            frame_list = os.listdir(pkl_path)
            sequence_len = len(frame_list)
            if sequence_len < 6:
                continue
            frame_list.sort(key=lambda x:int(x[3:-4]))
            self.sequence_list.append(dict(p=pkl_path, l=frame_list))
            self.index_map.append(self.index_map[-1] + sequence_len)

    def global_to_sequence_index(self, global_idx: int) -> Tuple[int, int]:
        for sequence_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[sequence_idx], self.index_map[sequence_idx+1]):
                frame_idx = global_idx - self.index_map[sequence_idx]
                # avoid out of range error
                return sequence_idx, frame_idx if frame_idx >= self.clip_range else self.clip_range - 1
        raise IndexError

    def pad_data(self, data):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        return data[r, :]

    def load_data(self, frame_fname):
        with open(frame_fname, 'rb') as f:
            frame = pickle.load(f)
        arbe_pcl = frame['arbe']
        arbe_feature = frame["arbe_feature"]

        # param: pose, shape
        if frame["mesh_param"] is None:
            return None, None
        opti_pcl = frame['optitrack']
        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]
        # mesh_vtx = frame["mesh_param"]["vertices"]
        mesh_jnt = frame["mesh_param"]["joints"]

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
        sequence_idx, frame_idx = self.global_to_sequence_index(idx)
        sequence = self.sequence_list[sequence_idx]
        clip = []

        data, label = self.load_data(os.path.join(sequence['p'], sequence['l'][frame_idx]))

        while data is None:
            frame_idx = random.randint(self.clip_range-1, len(sequence['l'])-1)
            data, label = self.load_data(os.path.join(sequence['p'], sequence['l'][frame_idx]))

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.step_between_clips):
            # get xyz and features
            clip_data, clip_label = self.load_data(os.path.join(sequence['p'], sequence['l'][clip_id]))
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
        return clip, label, (sequence_idx, frame_idx)


class MMDataset(Dataset):
    def __init__(self, driver_path, frames_per_clip=5, step_between_clips=1, train=True, 
                normal_scale=1, output_dim=151, skip_head=0, skip_tail=0, data_device='arbe', test_data='test'):
        self.driver_path = driver_path
        # range of frame index in a clip
        self.step_between_clips = step_between_clips
        self.clip_range = frames_per_clip * step_between_clips
        self.train = train
        self.normal_scale = normal_scale
        self.output_dim = output_dim
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.data_device = data_device
        self.num_points = 1024 if self.data_device=='arbe' else 4096
        self.test_data = test_data
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0,]
        self.sequence_list = []
        sequences_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        # self.selected_dirs = ['2022-03-25_16-54-28']
        # self.selected_dirs = [p for p in os.listdir('/home/nesc525/drivers/2') if p[:10] == '2022-03-25']
        for d_path in map(str, self.driver_path.split(",")):
            sequences_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]

        # init result loader, reindex
        for path in sequences_paths:
            try:
                sequence_loader = PKLLoader(path, params=[dict(tag="pkl_data", ext=".pkl")])
            except Exception as e:
                print(e)
                continue
            for pkl_fname in sequence_loader.pkls:
                with open(pkl_fname, "rb") as f:
                    sequence_data = pickle.load(f, encoding='bytes')
                sequence_len = len(sequence_data) - self.skip_head
                if sequence_len > 6:
                    self.sequence_list.append(pkl_fname)
                    self.index_map.append(self.index_map[-1] + sequence_len)
    
    def global_to_sequence_index(self, global_idx: int) -> Tuple[int, int]:
        for sequence_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[sequence_idx], self.index_map[sequence_idx+1]):
                frame_idx = global_idx - self.index_map[sequence_idx]
                # avoid out of range error
                return sequence_idx, frame_idx if frame_idx >= self.clip_range else self.clip_range - 1
        raise IndexError

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        sequence_idx, frame_idx = self.global_to_sequence_index(idx)
        frame_idx += self.skip_head
        sequence_fname = self.sequence_list[sequence_idx]
        with open(sequence_fname, "rb") as f:
            sequence = pickle.load(f, encoding='bytes')
        
        clip = []
        data = sequence[frame_idx]

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.step_between_clips):
            # get xyz and features
            clip_data = sequence[clip_id]
            # clip padding
            clip.append(clip_data[self.data_device])
        clip.append(data[self.data_device])
        clip = np.asarray(clip, dtype=np.float32)
        label = np.asarray(data['mesh_param'], dtype=np.float32)
        if True in np.isnan(clip):
            label = np.nan_to_num(clip)
        if True in np.isnan(label):
            label = np.nan_to_num(label)
            
        return clip, label, (sequence_idx, frame_idx)