import os
import time
from typing import Tuple
import numpy as np
import random
import pickle
from torch.utils.data import Dataset

from dataloader.result_loader import PKLLoader_test, ResultFileLoader, PKLLoader
from visualization.utils import pcl_filter, pcl_filter_nb, pcl_filter_nb_noground

class DepthMesh3D(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=4096,
            train=True, normal_scale = 1, output_dim=158, skip_head=0, skip_tail=0, device="sub1"):
        super(DepthMesh3D, self).__init__()
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
        #choice_dir = ['2021-10-18_18-50-29_M','2021-10-20_14-36-51_F'] #indoor
        #choice_dir = ['2021-10-22_10-42-50_O','2021-10-22_14-23-59_O'] #corridor
        #choice_dir = ['2021-10-22_16-56-25_O'] #outdoor
        #choice_dir = ['2021-10-22_17-29-59_O','2021-10-22_17-37-01_O'] #rain
        #choice_dir = ['2021-10-22_17-38-49_O','2021-10-22_17-49-03_O'] #smoke
        choice_dir =['2021-10-18_18-50-29_M']
        #choice_dir = ['2021-10-22_18-01-40_E','2021-10-22_18-14-58_O'] #night
        for root_path in map(str, self.root_path.split(",")):
            if self.train:
                videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if  p in choice_dir]
                #videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'T']
            else:
                #videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'M' or p[-1] == 'F']
                videos_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if  p in choice_dir]

        for idx, path in enumerate(videos_paths):
            # init result loader, reindex
            try:
                #video_loader = ResultFileLoader(root_path=path, skip_head=self.skip_head, skip_tail=500, select_key="mesh",enabled_sources=["sub1", "kinect_color","kinect_pcl", "kinect_depth","mesh", "mesh_param"])
                video_loader = ResultFileLoader(root_path=path, skip_head=self.skip_head, skip_tail=500, select_key="mesh",enabled_sources=["sub1", "kinect_color","kinect_pcl","mesh", "mesh_param"])
            except Exception as e:
                print(e)
                continue
            video_loader.skip_head = 500
            video_len = len(video_loader)
            #video_len = min(video_len, 300)
            self.index_map.append(self.index_map[-1] + video_len)
            self.video_loaders.append(video_loader)


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

    def load_data(self, video_loader, id):
        frame, info = video_loader[id]
        kinect_pcl = frame["sub1_pcl"]
        kinect_color = frame["sub1_color"]
        kinect_color = kinect_color.reshape(len(kinect_pcl), 3)
        kinect_data = np.hstack((kinect_pcl, kinect_color))
        # param: pose, shape
        if frame["mesh_param"] is None:
            return None, None, None
        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]
        mesh_vtx = frame["mesh_param"]["vertices"]
        # mesh_jnt = frame["mesh_jnt"]["keypoints"]
        # 0 for female, 1 for male
        gender = 1 if frame["information"].get("gender", "male") == "male" else 0
        bbox_center = ((mesh_vtx.max(axis=0) + mesh_vtx.min(axis=0))/2)[:3]
        # delete the zeros points
        # kinect_data = kinect_data[kinect_pcl.any(1)]
        # filter arbe_pcl with optitrack bounding box
        # print(mesh_vtx)
        kinect_data = pcl_filter(mesh_vtx, kinect_data, 0.2, 0.21)
        if kinect_data.shape[0] == 0:
            # remove bad frame
            return None, None, None

        # normalization
        kinect_data[:,:3] = (kinect_data[:,:3] - bbox_center)/self.normal_scale
        #RGB normalization?
        kinect_data[:,3:] /= np.array([256, 256, 256])
        # padding
        kinect_data = self.pad_data(kinect_data)
        mesh_pose[:3] += bbox_center
        label = np.concatenate((mesh_pose / self.normal_scale, mesh_shape / self.normal_scale, np.asarray(gender).reshape(-1)), axis=0)
        return kinect_data, label, info

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        video_idx, frame_idx = self.global_to_video_index(idx)
        video_loader = self.video_loaders[video_idx]
        clip = []

        data, label, info = self.load_data(video_loader, frame_idx)

        while data is None:
            frame_idx = random.randint(self.clip_range-1, len(video_loader)-1)
            data, label, info = self.load_data(video_loader, frame_idx)

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.step_between_clips):
            # get xyz and features
            clip_data, clip_label, _ = self.load_data(video_loader, clip_id)
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
        return clip, label, (video_idx, frame_idx, info)


class DepthMesh3D2(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=4096,
            train=True, normal_scale = 1, output_dim=158, skip_head=0, skip_tail=0, device="master"):
        super(DepthMesh3D2, self).__init__()
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
        self.device = device
        # self.pkl_path = [['2021-10-20_20-09-20_T','2021-10-20_20-06-52_T','2021-10-20_19-32-29_T',
        #                 '2021-10-20_19-24-07_T','2021-10-20_14-08-40_T','2021-10-20_14-06-35_T']]
        self.init_data()
        
    def init_data(self):
        video_paths = []

        for root_path in map(str, self.root_path.split(",")):
            #choice_dir = ['2021-10-18_18-50-29_M','2021-10-20_14-36-51_F'] #indoor
            #choice_dir = ['2021-10-22_10-46-57_E','2021-10-22_14-38-16_E'] #corridor
            #choice_dir = ['2021-10-22_16-56-25_T'] #outdoor
            #choice_dir = ['2021-10-22_17-29-59_T','2021-10-22_17-37-01_T'] #rain
            #choice_dir = ['2021-10-22_17-38-49_T','2021-10-22_17-49-03_T'] #smoke
            #choice_dir = ['2021-10-22_18-01-40_T','2021-10-22_18-14-58_T'] #night
            choice_dir = ['2021-10-22_18-01-40_T']
            if self.train:
                #video_paths += [os.path.join(root_path, p) for p in os.listdir(root_path)]
                video_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'T' ]
            else:
                video_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if  p in choice_dir]
                #video_paths += [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'F' or p[-1] == 'M']

        for p in video_paths:
            try:
                pkl_loader = PKLLoader_test(result_path=p, device=self.device)
                #pkl_loader = PKLLoader(result_path=p, device=self.device)
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
        return clip, label, (0,0,0)


