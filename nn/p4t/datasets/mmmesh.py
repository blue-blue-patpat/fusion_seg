import os
import numpy as np
import random
from torch.utils.data import Dataset

from dataloader.result_loader import ResultFileLoader
from visualization.utils import pcl_filter

class MMMesh3D(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=1024,
            train=True, normal_scale = 1.2, output_dim=111, skip_head=200, skip_tail=0):
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
        self.index_map = []
        self.video_loaders = []
        self.id_list = []

        if self.train:
            videos_path = [os.path.join(self.root_path, p) for p in os.listdir(self.root_path) if p[-1] == 'T']
        else:
            videos_path = [os.path.join(self.root_path, p) for p in os.listdir(self.root_path) if p[-1] == 'E']

        for idx, path in enumerate(videos_path):
            # init result loader
            video_loader = ResultFileLoader(root_path=path, enabled_sources=["arbe", "mesh", "mesh_param", "mesh_vtx_jnt"])
            # add video to list
            self.video_loaders.append(video_loader)
            # remove deduplicated arbe frames and head frames
            arbe_ids = sorted(map(int, video_loader.a_loader.file_dict["arbe"].drop_duplicates(subset=["dt"],keep="first")["id"]))[self.skip_head:-self.skip_tail-1]
            # add valid ids to list
            self.id_list.append(arbe_ids)
            # init index_map
            for i, id in enumerate(arbe_ids):
                if i > self.clip_range-1:
                    self.index_map.append((idx, id))

    def pad_data(self, data):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        return data[r, :]

    def get_data(self, video_loader, idx):
        frame, info = video_loader[idx]
        arbe_pcl = np.load(video_loader.a_loader[idx]["arbe"]["filepath"])[:,[0,1,2,3,7,8]]

        # param: pose, shape
        mesh_param = frame["mesh_param"]
        mesh_vtx = frame["mesh_vtx"]
        mesh_jnt = frame["mesh_jnt"]
        mesh_R = frame["mesh_R"]
        mesh_t = frame["mesh_t"]
        mesh_scale = frame["mesh_scale"]

        center = (mesh_vtx.max(axis=0) + mesh_vtx.min(axis=0))/2

        # filter radar_pcl with optitrack bounding box
        data = pcl_filter(mesh_vtx, arbe_pcl, 0.2)
        if data.shape[0] < 50:
            # remove bad frame
            data = None
        else:
            # normalization
            data[:,:3] = (data[:,:3] - center)/self.normal_scale
            data[:,3] /= 5e-38
            data[:,4] /= 5
            data[:,5] /= 150
            # padding
            data = self.pad_data(data)

        label = dict(
            param=mesh_param, vtx=mesh_vtx, jnt=mesh_jnt, R=mesh_R, t=mesh_t, sacle=mesh_scale
        )
        return data, label

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        video_idx, frame_idx = self.index_map[idx]
        video_loader = self.video_loaders[video_idx]
        ids = self.id_list[video_idx]
        clip = []
        i = ids.index(frame_idx)

        while True:
            data, label = self.get_data(video_loader, frame_idx)
            if data is None:
                i = random.randint(4, len(ids)-1)
                frame_idx = ids[i]
            else:
                break

        clip_ids = ids[i-self.clip_range+1:i:self.step_between_clips]
        for clip_id in clip_ids:
            # get xyz and features
            clip_data, clip_label = self.get_data(video_loader, clip_id)
            # remove bad frame
            if clip_data is None:
                clip_data = data
            # padding
            clip.append(clip_data)
        clip.append(data)

        clip = np.asarray(clip, dtype=np.float32)
        xyz_clip = clip[:,:,:3]
        feature_clip = clip[:,:,3:6].reshape(-1, 3, self.num_points)
        
        if True in np.isnan(label):
            label = np.nan_to_num(label)
        return xyz_clip, feature_clip, label, video_idx
