import os
import numpy as np
import random
from torch.utils.data import Dataset

from dataloader.result_loader import ResultFileLoader
from visualization.utils import pcl_filter

class DepthMesh(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=1024,
            train=True, normal_scale = 1.2, output_dim=95, skip_head=100, skip_tail=100):
        super(DepthMesh, self).__init__()
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

        if self.train:
            videos_path = [os.path.join(self.root_path, p) for p in os.listdir(self.root_path) if p[-1] == 'T']
        else:
            videos_path = [os.path.join(self.root_path, p) for p in os.listdir(self.root_path) if p[-1] == 'E']

        for path in videos_path:
            # init result loader, reindex
            video_loader = ResultFileLoader(root_path=path, enabled_sources=["arbe", "master", "sub1", "sub2", "kinect_pcl", "mesh", "mesh_param", "reindex"])
            # add video to list
            self.video_loaders.append(video_loader)
            self.index_map.append(self.index_map[-1] + len(video_loader))

    def pad_data(self, data):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        return data[r, :]

    def global_to_video_index(self, global_idx: int):
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
        data = np.vstack((arbe_pcl, arbe_feature))

        # param: pose, shape
        mesh_pose = frame["mesh_param"]["pose"]
        mesh_shape = frame["mesh_param"]["shape"]

        mesh_R = frame["mesh_R"]
        mesh_t = frame["mesh_t"]
        mesh_scale = frame["mesh_scale"]

        mesh_vtx = (frame["mesh_param"]["vertices"] @ mesh_R + mesh_t) * mesh_scale
        # mesh_jnt = frame["mesh_jnt"]["keypoints"]

        center = (mesh_vtx.max(axis=0) + mesh_vtx.min(axis=0))/2

        # filter radar_pcl with optitrack bounding box
        pcl_filtered = pcl_filter(mesh_vtx, data, 0.2)
        if pcl_filtered.shape[0] < 50:
            # remove bad frame
            data = None
        else:
            # normalization
            data = (pcl_filtered - center)/self.normal_scale
            arbe_feature /= np.array([5e-38, 5, 150])
            data = np.vstack((data, arbe_feature))
            # padding
            data = self.pad_data(data)

        label = np.concatenate((mesh_pose, mesh_shape, mesh_R.reshape(-1), mesh_t, mesh_scale.reshape(-1)), axis=0)
        return data, label

    def __len__(self):
        return len(self.index_map)

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
