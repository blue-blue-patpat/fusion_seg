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
    
    def process_image(self, image, joints=None, trans_mat=None, need_crop=False):
        if need_crop:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop person area using joints
            image, box_min, box_max = crop_image(joints, image, trans_mat, square=True, return_box=need_crop)
            image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype('float32'), (2,0,1))/255.0
        image = torch.from_numpy(image).float()
        transformed_img = self.normalize_img(image)
        if need_crop:
            return transformed_img, box_min, box_max
        return transformed_img
    
    def process_pcl(self, pcl, joints, padding_points=1024, mask=False):
        # filter person pcl using joints
        pcl = filter_pcl(joints, pcl)
        if not pcl.shape[0]:
            return torch.zeros((padding_points, 6)).float()
        if mask:
            # mask points
            indices = np.random.choice(pcl.shape[0], 100)
            pcl = pcl[indices]
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
        joints = mesh['joints'][:22]
        pose[:3] -= joints[0]
        
        # process image
        mas_img = torch.Tensor([])
        bbox = {}
        trans_mat_mas = seq_loader.calib['kinect_master']
        if 'master_image' in self.args.inputs:
            if self.args.read_orig_img:
                mas_img, box_min, box_max = self.process_image(frame['master_image'], joints, trans_mat_mas, True)
                bbox.update({'master_image':np.hstack((box_min, box_max))})
            else:
                mas_img = self.process_image(frame['master_image'], joints, trans_mat_mas)
                bbox.update({'master_image':frame['master_bbox'].reshape(-1)})

        sub_img = torch.Tensor([])
        trans_mat_sub = seq_loader.calib['kinect_sub']
        if 'sub_image' in self.args.inputs:
            if self.args.read_orig_img:
                sub_img, box_min, box_max = self.process_image(frame['sub_image'], joints, trans_mat_sub, True)
                bbox.update({'sub_image':np.hstack((box_min, box_max))})
            else:
                sub_img = self.process_image(frame['sub_image'], joints, trans_mat_sub)
                bbox.update({'sub_image':frame['sub_bbox'].reshape(-1)})
               
        # process radar pcl
        radar_pcl = torch.Tensor([])
        if 'radar' in self.args.inputs:
            radar_pcl = frame['radar']
            radar_pcl[:,3:] /= np.array([5e-38, 5., 150.])
            radar_pcl = self.process_pcl(radar_pcl, joints, padding_points=1024)
            
        # process depth pcl
        if self.args.train:
            random_value = np.random.rand()
            mask = True if random_value >= 0.7 else False
        else:
            mask = False
        mas_depth = torch.Tensor([])
        if 'master_depth' in self.args.inputs:
            mas_depth = self.process_pcl(frame['master_depth'], joints, padding_points=4096, mask=mask)
        
        sub_depth = torch.Tensor([])
        if 'sub_depth' in self.args.inputs:
            sub_depth = self.process_pcl(frame['sub_depth'], joints, padding_points=4096, mask=mask)
             
        # transform matrix
        trans_mat_mas = torch.tensor(trans_mat_2_tensor(trans_mat_mas)).float()
        trans_mat_sub = torch.tensor(trans_mat_2_tensor(trans_mat_sub)).float()
        trans_mat = {
            'radar': torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).float(),
            'master_image': trans_mat_mas,
            'master_depth': trans_mat_mas,
            'sub_image': trans_mat_sub,
            'sub_depth': trans_mat_sub,
        }
        
        # return result
        result = {}
        result['radar'] = radar_pcl
        result['master_image'] = mas_img
        result['master_depth'] = mas_depth
        result['sub_image'] = sub_img
        result['sub_depth'] = sub_depth
        result['trans_mat'] = trans_mat
        result['bbox'] = bbox
        result['root_pelvis'] = torch.from_numpy(joints[0]).float()

        label = np.concatenate((pose, betas), dtype=np.float32, axis=0)
        label = torch.from_numpy(label).float()
        
        return result, label

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader =self.full_data[seq_path]
        
        return self.load_data(seq_loader, frame_idx)