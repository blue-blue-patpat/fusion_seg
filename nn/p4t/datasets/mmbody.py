import os
import numpy as np
import random
from torch.utils.data import Dataset

from dataloader.result_loader import ResultFileLoader
from visualization.utils import pcl_filter

class MMBody3D(Dataset):
    def __init__(self, root_path, frames_per_clip=5, step_between_clips=1, num_points=1024,
            train=True, normal_scale = 1.2, output_dim=111, skip_head=200, skip_tail=0):
        super(MMBody3D, self).__init__()
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
            video_loader = ResultFileLoader(root_path=path, enabled_sources=["arbe", "arbe_feature", "optitrack"])
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

    def get_data(self, video_loader, id):
        frame, info = video_loader[id]
        arbe_pcl = frame["arbe"]
        arbe_feature = frame["arbe_feature"][:, [0,4,5]]
        arbe_data = np.hstack((arbe_pcl, arbe_feature))
        label = frame["optitrack"]
        center = (label.max(axis=0) + label.min(axis=0))/2
        # filter arbe_pcl with optitrack bounding box
        arbe_data = pcl_filter(label, arbe_data, 0.2)
        if arbe_data.shape[0] < 50:
            # remove bad frame
            arbe_data = None
        else:
            # normalization
            arbe_data[:,:3] = (arbe_data[:,:3] - center)/self.normal_scale
            arbe_data[:,3:] /= np.array([5e-38, 5, 150])
            # padding
            arbe_data = self.pad_data(arbe_data)
        label = np.asarray(((label - center) / self.normal_scale).reshape(-1), dtype=np.float32)
        return arbe_data, label

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, id = self.index_map[idx]
        video_loader = self.video_loaders[index]
        ids = self.id_list[index]
        clip = []
        clip_label = []
        i = ids.index(id)

        while True:
            anchor_frame, label = self.get_data(video_loader, id)
            if anchor_frame is None:
                i = random.randint(self.clip_range-1, len(ids)-1)
                id = ids[i]
            else:
                break

        clip_ids = ids[i-self.clip_range+1:i:self.step_between_clips]
        for clip_id in clip_ids:
            # get xyz and features
            clip_frame, frame_label = self.get_data(video_loader, clip_id)
            # remove bad frame
            if clip_frame is None:
                clip_frame = anchor_frame
            # padding
            clip.append(clip_frame)
            clip_label.append(frame_label)
        clip.append(anchor_frame)
        clip_label.append(label)

        clip = np.asarray(clip, dtype=np.float32)
        clip_label = np.asarray(clip_label, dtype=np.float32)

        return clip, clip_label, self.index_map[idx]
