# author:Leo-Zhu  time:21-8-31

import os
import numpy as np
from torch.utils.data import Dataset
from dataloader.result_loader import KinectResultLoader, ArbeResultLoader, OptitrackResultLoader


class DataLoader(Dataset):
    def __init__(self, root_path, frames_per_clip=2, step_between_clips=1, num_points=2048, train=True):
        super(DataLoader, self).__init__()

        self.clip = []
        self.labels = []
        # init result loader
        self.arbe_loader = ArbeResultLoader(root_path)
        self.opti_loader = OptitrackResultLoader(root_path)
        # remove deduplicated arbe frames
        self.arbe_loader.file_dict['arbe'] = self.arbe_loader.file_dict['arbe'].drop_duplicates(subset=['dt'],keep='first')

        self.index_map = []

        file_num = len(self.arbe_loader)
        for i in range(file_num):
            if i > frames_per_clip*step_between_clips-1 and i < file_num-frames_per_clip*step_between_clips+1:
                for f in range(i-frames_per_clip*step_between_clips, i+frames_per_clip*step_between_clips+1, step_between_clips):
                    self.index_map.append((i, f))

        # for label_name in os.listdir(out_root):
        #     label = np.load(os.path.join(out_root, label_name), allow_pickle=True)
        #     self.labels.append(label)

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        index, t = self.index_map[idx]
        video = self.videos[index]
        label = self.labels[index]

        clip = [self.videos[t+i*self.step_between_clips] for i in range(self.frames_per_clip*2+1)]
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


if __name__ == '__main__':
    dataset_1 = DataLoader(root_path='/home/leo/in_root', out_root='/home/leo/out_root')
    clip, label, video_idx = dataset_1[0]
    print(clip)
    print(label)
    print(video_idx)
