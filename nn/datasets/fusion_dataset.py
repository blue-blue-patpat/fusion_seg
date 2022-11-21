import cv2
import os.path as op
import numpy as np

import torch
import torchvision.transforms as transforms
from nn.datasets.utils import *
    
class FusionDataset():
    def __init__(self, args, **kwargs):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.seq_idxes = eval(args.seq_idxes) if args.seq_idxes else range(20)
        self.args = args
        self.resource = args.inputs
        self.init_index_map()
            
    def init_index_map(self):
        self.index_map = [0,]
        if self.args.train:
            seq_dirs = ['sequence_{}'.format(i) for i in self.seq_idxes]
            self.seq_paths = [op.join(self.data_path, p) for p in seq_dirs]
        else:
            seq_dirs = ['sequence_{}'.format(i) for i in range(2)]
            self.seq_paths = [op.join(self.data_path, self.args.test_scene, p) for p in seq_dirs]
        
        print('Data path: ', self.seq_paths)

        self.seq_loaders = []
        self.full_data = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = SequenceLoader(path, self.args.skip_head, resource=self.resource)
            self.full_data.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
    def global_to_seq_index(self, global_idx:int):
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx
        raise IndexError
    
    def process_image(self, image, joints=None, need_crop=False):
        if need_crop:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop person area using joints
            image = crop_image(joints, image, square=True)
            image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype('float32'), (2,0,1))/255.0
        image = torch.from_numpy(image).float()
        transformed_img = self.normalize_img(image)
        return transformed_img
    
    def process_pcl(self, pcl, joints, trans_mat, padding_points=1024, mask=False):
        # filter person pcl using joints
        pcl = filter_pcl(joints, pcl)
        if not pcl.shape[0]:
            return torch.zeros((padding_points, 6)).float()
        if mask:
            # mask points
            indices = np.random.choice(pcl.shape[0], 100)
            pcl = pcl[indices]
        if trans_mat is not None:
            trans_pcl = (pcl[:,:3] - trans_mat['t']) @ trans_mat['R']
            pcl = np.hstack((trans_pcl, pcl[:,3:]))
        # normalize pcl
        pcl[:,:3] -= joints[0]
        # padding pcl
        pcl = pad_pcl(pcl, padding_points)
        pcl = torch.from_numpy(pcl).float()
        return pcl
    
    def load_data(self, seq_loader, frame_idx):
        if isinstance(seq_loader, list):
            return seq_loader[frame_idx]
        frame = seq_loader[frame_idx]
        # get mesh parameters
        mesh = dict(frame['mesh'])
        pose = mesh['pose']
        betas = mesh['shape']
        gender = 'None'
        joints = mesh['joints'][:22]
            
        # process image
        mas_img = torch.Tensor([])
        trans_mat_mas = seq_loader.calib['kinect_master']
        if 'master_image' in self.args.inputs:
            joints_mas = (joints - trans_mat_mas['t']) @ trans_mat_mas['R']
            mas_img = self.process_image(frame['master_image'], joints_mas)

        sub_img = torch.Tensor([])
        trans_mat_sub = seq_loader.calib['kinect_sub']
        if 'sub_image' in self.args.inputs:
            joints_sub = (joints - trans_mat_sub['t']) @ trans_mat_sub['R']
            sub_img = self.process_image(frame['sub_image'], joints_sub)
                
        # transform mesh param and joints to the kinect coordinate if need
        trans_pose = pose
        trans_pose[:3] -= joints[0]
        trans_mat_to_cam = None
                
        # process radar pcl
        radar_pcl = torch.Tensor([])
        if 'radar' in self.args.inputs:
            radar_pcl = frame['radar']
            radar_pcl[:,3:] /= np.array([5e-38, 5., 150.])
            radar_pcl = self.process_pcl(radar_pcl, joints, trans_mat_to_cam, padding_points=1024)
            
        # process depth pcl
        if self.args.train:
            random_value = np.random.rand()
            mask = True if random_value >= 0.7 else False
        else:
            mask = False
        mas_depth = torch.Tensor([])
        if 'master_depth' in self.args.inputs:
            mas_depth = frame['master_depth']
            mas_depth = self.process_pcl(mas_depth, joints, trans_mat_to_cam, padding_points=4096, mask=mask)
        
        sub_depth = torch.Tensor([])
        if 'sub_depth' in self.args.inputs:
            sub_depth = frame['sub_depth']
            sub_depth = self.process_pcl(sub_depth, joints, trans_mat_to_cam, padding_points=4096, mask=mask)
        
        # return result
        result = {}
        result['radar'] = radar_pcl
        result['master_image'] = mas_img
        result['master_depth'] = mas_depth
        result['sub_image'] = sub_img
        result['sub_depth'] = sub_depth

        label = np.concatenate((trans_pose, betas), dtype=np.float32, axis=0)
        label = torch.from_numpy(label).float()
                
        return result, label

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader =self.full_data[seq_path]
        
        return self.load_data(seq_loader, frame_idx)