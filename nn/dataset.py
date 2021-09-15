# author:Leo-Zhu  time:21-8-31

import os
from posixpath import join
import numpy as np
from torch.utils.data import Dataset
from dataloader.result_loader import ResultFileLoader
from visualization.utils import pcl_filter


class MMBody3D(Dataset):
    def __init__(self, root_path, frames_per_clip=2, step_between_clips=1, num_points=2048, train=True, skip_head=100):
        super(MMBody3D, self).__init__()
        self.step_between_clips = step_between_clips
        # range of frame index in a clip
        self.clip_range = frames_per_clip * step_between_clips
        self.num_points = num_points
        self.train = train
        self.video_list = []
        self.index_map = []
        self.id_list = []
        videos = [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'D']
        for path in videos:
            # init result loader
            video_loader = ResultFileLoader(root_path=path, enabled_sources=["arbe", "optitrack"])
            # add video to list
            self.video_list.append(video_loader)

        for idx, video_res in enumerate(self.video_list):
            # remove deduplicated arbe frames and head frames
            arbe_ids = sorted(map(int, video_res.a_loader.file_dict["arbe"].drop_duplicates(subset=["dt"],keep="first")["id"]))[skip_head:]
            file_num = len(arbe_ids)
            # add validated ids of every video to list
            self.id_list.append(arbe_ids)
            # index_map for frames of all train data
            for i, id in enumerate(arbe_ids):
                if i > self.clip_range-1 and i < file_num-self.clip_range:
                    self.index_map.append((idx, id))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, id = self.index_map[idx]
        video = self.video_list[index]
        ids = self.id_list[index]
        clip = []
        label = video[id][0]["optitrack"]
        self.num_class = label.shape[0]
        i = ids.index(id)
        clip_ids = ids[i-self.clip_range:i+self.clip_range+1:self.step_between_clips] 
        for clip_id in clip_ids:
            frame = np.load(video.a_loader[clip_id]["arbe"]["filepath"])
            # filter arbe_pcl with optitrack bounding box
            arbe_pcl = pcl_filter(label, frame, 0.5)
            clip.append(arbe_pcl)

        for i, f in enumerate(clip):
            if f.shape[0] > self.num_points:
                r = np.random.choice(f.shape[0], size=self.num_points, replace=False)
            # padding the points
            else:
                repeat, residue = self.num_points // f.shape[0], self.num_points % f.shape[0]
                r = np.random.choice(f.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(f.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = f[r, :]
        clip = np.asarray(clip, dtype=np.float32)
        return clip, label
