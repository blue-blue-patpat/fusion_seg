# author:Leo-Zhu  time:21-8-31

import os
from posixpath import join
import numpy as np
from torch.utils.data import Dataset
from dataloader.result_loader import ResultFileLoader


class DataLoader(Dataset):
    def __init__(self, root_path, frames_per_clip=2, step_between_clips=1, num_points=2048, train=True, skip_head=30):
        super(DataLoader, self).__init__()
        self.step_between_clips = step_between_clips
        self.clip_range = frames_per_clip * step_between_clips
        self.num_points = num_points
        self.train = train
        self.file_res = []
        self.index_map = []
        self.id_map = []
        videos = [os.path.join(root_path, p) for p in os.listdir(root_path) if p[-1] == 'D']
        for path in videos:
            # init result loader
            file_loader = ResultFileLoader(root_path=path, skip_head=skip_head, enabled_sources=["arbe", "optitrack"])
            self.file_res.append(file_loader)

        for idx, file_res in enumerate(self.file_res):
            # remove deduplicated arbe frames
            arbe_ids = sorted(map(int, file_res.a_loader.file_dict["arbe"].drop_duplicates(subset=["dt"],keep="first")["id"]))[skip_head:]
            file_num = len(arbe_ids)
            self.id_map.append(arbe_ids)
            for i, id in enumerate(arbe_ids):
                if i > self.clip_range-1 and i < file_num-self.clip_range:
                    self.index_map.append((idx, id))
        pass
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, id = self.index_map[idx]
        arbe_video = self.file_res[index]
        ids = self.id_map[index]
        clip = []
        for i, item in enumerate(ids):
            if item == id:
                clip_ids = ids[i-self.clip_range:i+self.clip_range+1:self.step_between_clips] 
                clip.append(np.load(arbe_video.a_loader[clip_id]["arbe"]["filepath"]) for clip_id in clip_ids)
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        return clip.astype(np.float32), label, index


if __name__ == "__main__":
    dataset_1 = DataLoader(root_path="/home/leo/in_root", out_root="/home/leo/out_root")
    clip, label, video_idx = dataset_1[0]
    print(clip)
    print(label)
    print(video_idx)
