import os
from typing import Tuple
import numpy as np
import random
from torch.utils.data import Dataset
import pickle

from dataloader.result_loader import ResultFileLoader, PKLLoader
from visualization.utils import pcl_filter


class MMMesh3D(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=1024,
            train=True, normal_scale = 1, output_dim=158, skip_head=0, skip_tail=0):
        super(MMMesh3D, self).__init__()
        self.root_path = root_path
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
        self.video_loaders = []
        videos_paths = []
        self.sequence_path = []
        choice_dir = ['2021-10-22_18-01-40_N']
        for root_path in map(str, self.root_path.split(",")):
            if self.train:
                videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p in choice_dir]
                # videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'T']
                #videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path)]
            else:
                videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p in choice_dir]
                # videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'F' or p[-1] == 'M']
                # videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'T']

        for idx, path in enumerate(videos_paths):
            # init result loader, reindex
            try:
                video_loader = ResultFileLoader(root_path=path, skip_head=self.skip_head, skip_tail=60, enabled_sources=["arbe", "arbe_feature", "mesh", "mesh_param", "reindex"])
            except Exception as e:
                print(e)
                continue
            if len(video_loader) < 6:
                continue
            video_loader.skip_head = 300
            self.video_loaders.append(video_loader)
            self.sequence_path.append(path)
            video_len = len(video_loader)
            # if int(path[32:34]) >= 22:
            #     video_len = min(video_len, 300)
            self.index_map.append(self.index_map[-1] + video_len)

    def pad_data(self, data):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        return data[r, :]

    def global_to_video_index(self, global_idx: int) -> Tuple[int, int]:
        for video_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[video_idx], self.index_map[video_idx+1]):
                frame_idx = global_idx - self.index_map[video_idx]
                # avoid out of range error
                return video_idx, frame_idx if frame_idx >= self.clip_range else self.clip_range - 1
        raise IndexError

    def load_data(self, video_loader, idx):
        frame, info = video_loader[idx]
        arbe_pcl = frame["arbe"]
        arbe_feature = frame["arbe_feature"][:, [0,4,5]]
        
        # normalization
        arbe_feature /= np.array([5e-38, 5, 150])

        # param: pose, shape
        if frame["mesh_param"] is None:
            return None, None
        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]

        mesh_vtx = frame["mesh_param"]["vertices"]
        # mesh_jnt = frame["mesh_jnt"]["keypoints"]

        # 0 for female, 1 for male
        gender = 1 if frame["information"].get("gender", "male") == "male" else 0

        # filter radar_pcl with optitrack bounding box
        arbe_data = pcl_filter(mesh_vtx, np.hstack((arbe_pcl, arbe_feature)), 0.2)

        if arbe_data.shape[0] < 50:
            # remove bad frame
            return None, None

        bbox_center = ((mesh_vtx.max(axis=0) + mesh_vtx.min(axis=0))/2)[:3]
        arbe_data[:,:3] -= bbox_center
        mesh_pose[:3] += bbox_center

        # arbe_data = np.hstack((arbe_data, np.repeat([arbe_center], arbe_data.shape[0], axis=0)))

        # padding
        arbe_data = self.pad_data(arbe_data)

        label = np.concatenate((mesh_pose / self.normal_scale, mesh_shape / self.normal_scale, np.asarray(gender).reshape(-1)), axis=0)
        # label = np.concatenate((mesh_pose, mesh_shape), axis=0) / self.normal_scale
        return arbe_data, label

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        video_idx, frame_idx = self.global_to_video_index(idx)
        video_loader = self.video_loaders[video_idx]
        clip = []

        data, label = self.load_data(video_loader, frame_idx)

        while data is None:
            frame_idx = random.randint(self.clip_range-1, len(video_loader)-1)
            data, label = self.load_data(video_loader, frame_idx)

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.step_between_clips):
            # get xyz and features
            clip_data, clip_label = self.load_data(video_loader, clip_id)
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
        return clip, label, (video_idx, frame_idx)


class MMMeshPKL(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=4096,
            train=True, normal_scale = 1, output_dim=158, skip_head=0, skip_tail=0):
        super(MMMeshPKL, self).__init__()
        self.root_path = root_path
        # range of frame index in a clip
        self.step_between_clips = step_between_clips
        self.clip_range = frames_per_clip * step_between_clips
        self.num_points = num_points
        self.train = train
        self.normal_scale = normal_scale
        self.output_dim = output_dim
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.clips = []
        self.labels = []
        self.init_data()
        
    def init_data(self):
        video_paths = []
        for root_path in map(str, self.root_path.split(",")):
            if self.train:
                video_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1]=='T' or p[-1]=='O' or p[-1]=='N']
            else:
                video_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-3] == 'R']
                # video_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'M' or p[-1] == 'F']

        for p in video_paths:
            try:
                pkl_loader = PKLLoader(p, params = [dict(tag="mmWave_data", ext=".pkl")])
            except Exception as e:
                print(e)
                continue
            for pkl in pkl_loader.pkls:
                with open(pkl, "rb") as f:
                    if pkl[-10] == 'X':
                        self.clips += pickle.load(f, encoding='bytes')
                    elif pkl[-10] == 'y':
                        self.labels += pickle.load(f, encoding='bytes')
                    else:
                        raise RuntimeError("No pkl data")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]
        return clip, label, (0, 0)


# class MMMesh3D2(Dataset):
#     def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=1024,
#             train=True, normal_scale = 1, output_dim=66):
#         super(MMMesh3D2, self).__init__()
#         self.root_path = root_path
#         # range of frame index in a clip
#         self.step_between_clips = step_between_clips
#         self.clip_range = frames_per_clip * step_between_clips
#         self.num_points = num_points
#         self.train = train
#         self.normal_scale = normal_scale
#         self.output_dim = output_dim
#         self.init_index_map()

#     def init_index_map(self):
#         self.index_map = [0,]
#         self.video_loaders = []

#         if self.train:
#             videos_path = [os.path.join(self.root_path, p) for p in os.listdir(self.root_path) if p[-1] == 'T' or p[-1] == 'N']
#         else:
#             videos_path = [os.path.join(self.root_path, p) for p in os.listdir(self.root_path) if p[-1] == 'E']

#         for idx, path in enumerate(videos_path):
#             # init result loader, reindex
#             video_loader = ResultFileLoader(root_path=path, enabled_sources=["arbe", "arbe_feature", "mesh", "mesh_param", "reindex"])
#             # add video to list
#             self.video_loaders.append(video_loader)
#             self.index_map.append(self.index_map[-1] + len(video_loader))

#     def pad_data(self, data):
#         if data.shape[0] > self.num_points:
#             r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
#         else:
#             repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
#             r = np.random.choice(data.shape[0], size=residue, replace=False)
#             r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
#         return data[r, :]

#     def global_to_video_index(self, global_idx: int) -> Tuple[int, int]:
#         for video_idx in range(len(self.index_map)-1):
#             if global_idx in range(self.index_map[video_idx], self.index_map[video_idx+1]):
#                 frame_idx = global_idx - self.index_map[video_idx]
#                 # avoid out of range error
#                 return video_idx, frame_idx if frame_idx >= self.clip_range else self.clip_range - 1
#         raise IndexError

#     def load_data(self, video_loader, idx):
#         frame, info = video_loader[idx]
#         arbe_pcl = frame["arbe"]
#         arbe_feature = frame["arbe_feature"][:, [0,4,5]]
        
#         # normalization
#         arbe_feature /= np.array([5e-38, 5, 150])

#         # param: pose, shape
#         mesh_pose = frame["mesh_param"]["pose"]
#         mesh_shape = frame["mesh_param"]["shape"]

#         mesh_vtx = frame["mesh_param"]["vertices"]
#         # mesh_jnt = frame["mesh_jnt"]["keypoints"]

#         # filter radar_pcl with optitrack bounding box
#         arbe_data = pcl_filter(mesh_vtx, np.hstack((arbe_pcl, arbe_feature)), 0.2)

#         if arbe_data.shape[0] < 50:
#             # remove bad frame
#             return None, None

#         bbox_center = ((mesh_vtx.max(axis=0) + mesh_vtx.min(axis=0))/2)[:3]
#         arbe_data[:,:3] -= bbox_center
#         mesh_pose[:3] += bbox_center

#         # arbe_data = np.hstack((arbe_data, np.repeat([arbe_center], arbe_data.shape[0], axis=0)))

#         # padding
#         arbe_data = self.pad_data(arbe_data)

#         label = np.concatenate((mesh_pose, mesh_shape), axis=0) / self.normal_scale
#         return arbe_data, label

#     def __len__(self):
#         return self.index_map[-1]

#     def __getitem__(self, idx):
#         video_idx, frame_idx = self.global_to_video_index(idx)
#         video_loader = self.video_loaders[video_idx]
#         clip = []

#         data, label = self.load_data(video_loader, frame_idx)

#         while data is None:
#             frame_idx = random.randint(self.clip_range-1, len(video_loader)-1)
#             data, label = self.load_data(video_loader, frame_idx)

#         for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.step_between_clips):
#             # get xyz and features
#             clip_data, clip_label = self.load_data(video_loader, clip_id)
#             # remove bad frame
#             if clip_data is None:
#                 clip_data = data
#             # padding
#             clip.append(clip_data)
#         clip.append(data)

#         clip = np.asarray(clip, dtype=np.float32)
#         label = np.asarray(label, dtype=np.float32)
        
#         if True in np.isnan(label):
#             label = np.nan_to_num(label)
#         label = label[6:72]
#         return clip, label, (video_idx, frame_idx)