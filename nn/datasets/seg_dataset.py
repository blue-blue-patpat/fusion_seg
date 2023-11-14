import cv2
import os.path as op
import numpy as np
import io
import torch
import torchvision.transforms as transforms
from nn.datasets.utils import *
# from utils import *
from PIL import Image, ImageOps


class SegDataset:
    def __init__(self, args, **kwargs):
    # def __init__(self):
        self.args = args
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ###图像归一化参数
        self.img_res = 224 
        self.seq_idxes = range(1,11) ### 看一共有多少个sequence
        self.intrinsic = np.array([[2.4818480434445055e+03, 0., 8.6582666108041667e+02], 
		    			   [0., 2.0067972011730976e+03, 6.7647122567546967e+02], 
						   [0., 0., 1.]], dtype=np.float64)
        self.extrinsic = np.array([[-1.1253927874533565e-01, -9.7959951222695996e-01, 1.6649236133883782e-01, 1.3490402402365802e+00],
       					   [-1.2920108073891556e-01, -1.5170804023555751e-01, -9.7994476949661968e-01, -2.3870673832544825e-01],
       					   [9.8521164806115458e-01, -1.3179327058916540e-01, -1.0949220224306000e-01, 6.5199882603717707e-01], 
       					   [0., 0., 0., 1.]], dtype=np.float64)
        self.data_path = args.data_path
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0, ]
        if self.args.train:
            seq_dirs = ['seq_{}'.format(i) for i in self.seq_idxes]
            self.seq_paths = [op.join(self.data_path, p) for p in seq_dirs]
        # else:
        #     seq_dirs = ['seq_{}'.format(i) for i in range(2)]
        #     self.seq_paths = [op.join(self.data_path, self.args.test_scene, p) for p in seq_dirs]
        
        print('Data path: ', self.seq_paths)

        self.seq_loaders = []
        self.full_data = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = SequenceLoader2(path)
            self.full_data.update({path: seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))

    def global_to_seq_index(self, global_idx: int):
        for seq_idx in range(len(self.index_map) - 1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx + 1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx+1
        raise IndexError


    def process_input(self, image, pcd, label, padding_points=4096):  ####padding_points数目需要确定
        # if not pcd.shape[0]:
        #     return torch.zeros((padding_points, 6)).float()
        # process image
        image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype("float32"), (2, 0, 1)) / 255.0
        image = torch.from_numpy(image).float()
        image = self.normalize_img(image)
        # get origin image label
        img_ori_label = np.array(label)
        # project pcd to image
        point_in_image, project_mask = get_fov_mask(pcd, self.extrinsic, self.intrinsic, 1080, 1920)
        # get pcd label
        pcd_label = img_ori_label[point_in_image[:,1].astype(np.int32), point_in_image[:,0].astype(np.int32)]
        # get filter pcd 
        pcd = pcd[project_mask]
        # padding pcd and pcd label
        pcd, random_mask = pad_pcd(pcd, padding_points)
        pcd_label = pcd_label[random_mask]
        # get pcd and pcd label after padding
        pcd = torch.from_numpy(pcd).float()
        pcd_label = torch.from_numpy(pcd_label).to(torch.int32)
        # get image label
        img_label = label.resize((self.img_res, self.img_res), resample=Image.BILINEAR)
        img_label = np.array(img_label)
        img_label = torch.from_numpy(img_label).to(torch.int32)
        return image, img_label, pcd, pcd_label

    def load_data(self, seq_loader, frame_idx):
        if isinstance(seq_loader, list):
            return seq_loader[frame_idx]
        frame = seq_loader[frame_idx]

        # preprocess lidar pcd
        pcd = frame["lidar"]
        pcd = pcd[~np.isnan(pcd).any(axis=1)] # remove points of nan
        # process input
        img, img_label, pcd, pcd_label = self.process_input(frame['image'], pcd, frame['label'])

        # calibration
        trans_mat = {
            'points': torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).float(),
            'images': torch.tensor(self.extrinsic).float()
        }

        # return result
        result = {}
        result["lidar"] = pcd
        result["image"] = img
        result["trans_mat"] = trans_mat
        result['img_label'] = img_label
        result['pcd_label'] = pcd_label

        return result

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader = self.full_data[seq_path]
        return self.load_data(seq_loader, frame_idx)
    

