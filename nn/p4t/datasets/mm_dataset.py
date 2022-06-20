import os
from typing import Tuple
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
from multiprocessing import Process

from dataloader.result_loader import ResultFileLoader, PKLLoader
from visualization.utils import pcl_filter, get_rgb_feature, image_crop, pcl_project
from nn.p4t.datasets.folder_list import *

class mmWave(Dataset):
    def __init__(self, driver_path, train=True, **kwargs):
        self.driver_path = driver_path
        self.train = train
        # range of frame index in a clip
        self.clip_step = kwargs.get('clip_step', 1)
        self.clip_frames = kwargs.get('clip_frames', 5)
        self.clip_range = self.clip_frames * self.clip_step
        self.output_dim = kwargs.get('output_dim', 151)
        self.skip_head = kwargs.get('skip_head', 0)
        self.skip_tail = kwargs.get('skip_tail', 0)
        self.test_data = kwargs.get('test_data', 'lab1')
        self.use_pkl = kwargs.get('use_pkl', True)
        self.create_pkl = kwargs.get('create_pkl', True)
        self.enable_sources = kwargs.get('enable_sources',["arbe","arbe_feature","master","kinect_color","mesh","mosh","mesh_param"])
        self.input_data = self.__class__.__name__
        self.num_points = kwargs.get('num_points', 1024)
        self.feature_type = kwargs.get('feature_type', 'arbe')
        self.features = kwargs.get('features', 3)
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = []
        seq_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        for d_path in map(str, self.driver_path.split(',')):
            seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]

        process_dict = {}
        self.full_data = {}
        for path in seq_paths:
            seq_pkl = os.path.join(path, 'pkl_data/{}_{}.pkl'.format(self.input_data, self.feature_type))
            try:
                if os.path.exists(seq_pkl) and self.use_pkl:
                    with open(seq_pkl, 'rb') as f:
                        seq_loader = pickle.load(f)[self.skip_head:-self.skip_tail-1]
                else:
                    # init result loader, reindex
                    seq_loader = ResultFileLoader(path, self.skip_head, self.skip_tail, self.enable_sources)
                    if self.create_pkl:
                        p = Process(target=self.write_pkl, args=(path, seq_pkl, seq_loader))
                        p.start()
                        process_dict.update({path: p})
            except Exception as e:
                print(e)
                continue

            if len(seq_loader) < 6:
                continue
            self.full_data.update({path:seq_loader})
            self.seq_paths.append(path)
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
        if self.create_pkl:
            for path, p in process_dict.items():
                p.join()
                with open(os.path.join(path, 'pkl_data/{}_{}.pkl'.format(self.input_data, self.feature_type)), 'rb') as f:
                    self.full_data[path] = pickle.load(f)

    def write_pkl(self, path, pkl_fname, seq_loader):
        import time
        data = []
        for i in range(len(seq_loader)):
            t_s = time.time()
            data.append(self.load_data(seq_loader, i))
            t_e = time.time()
            print(path, i, t_e-t_s, 's')
        if not os.path.exists(os.path.join(path, 'pkl_data')):
            os.mkdir(os.path.join(path, 'pkl_data'))
        with open(pkl_fname, 'wb') as f:
            pickle.dump(data, f)
        print(path, 'Done')

    def pad_data(self, data, return_choices=False):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        if return_choices:
            return data[r, :], r
        return data[r, :]

    def global_to_seq_index(self, global_idx: int) -> Tuple[int, int]:
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                # avoid out of range error
                return seq_idx, frame_idx if frame_idx >= self.clip_range else self.clip_range - 1
        raise IndexError

    def __len__(self):
        return self.index_map[-1]

    def load_data(self, seq_loader, idx):
        if isinstance(seq_loader, list):
            return seq_loader[idx]

        frame, _ = seq_loader[idx]
        arbe_pcl = frame['arbe']
        arbe_feature = frame['arbe_feature'][:, [0,4,5]]
        arbe_feature /= np.array([5e-38, 5, 150])
        rgb_data = frame['master_color']
        trans_mat = seq_loader.trans['kinect_master']

        # param: pose, shape
        if frame['mesh_param'] is None:
            return None, None
        
        mesh_pose = frame['mesh_param']['pose']
        mesh_shape = frame['mesh_param']['shape']
        mesh_joint = frame['mesh_param']['joints']

        if self.feature_type == 'arbe':
            # filter radar_pcl with bounding box
            arbe_data = pcl_filter(mesh_joint, np.hstack((arbe_pcl, arbe_feature)), 0.2)
        
        elif self.feature_type == 'rgb':
            arbe_data = pcl_filter(mesh_joint, arbe_pcl, 0.2)
            # transform radar pcl coordinate to kinect master
            trans_pcl = (arbe_data - trans_mat['t']) @ trans_mat['R']
            arbe_data = get_rgb_feature(trans_pcl, rgb_data, visual=False)

        elif self.feature_type == 'arbe_and_rgb':
            arbe_data = pcl_filter(mesh_joint, np.hstack((arbe_pcl, arbe_feature)), 0.2)
            trans_pcl = (arbe_data[:, :3] - trans_mat['t']) @ trans_mat['R']
            rgb_feature = get_rgb_feature(trans_pcl, rgb_data, visual=False)
            arbe_data = np.hstack((arbe_data, rgb_feature[:, 3:]))

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None

        bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0))/2)[:3]
        arbe_data[:,:3] -= bbox_center
        mesh_pose[:3] -= bbox_center

        # padding
        arbe_data = self.pad_data(arbe_data)
        label = np.concatenate((mesh_pose, mesh_shape), axis=0)

        return arbe_data, label

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        clip = []
        seq_loader =self.full_data[seq_path]
        data, label = self.load_data(seq_loader, frame_idx)

        while data is None:
            frame_idx = random.randint(self.clip_range-1, len(seq_loader)-1)
            data, label = self.load_data(seq_loader, frame_idx)

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.clip_step):
            # get xyz and features
            clip_data, _ = self.load_data(seq_loader, clip_id)
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

        return clip, label, (seq_idx, frame_idx)


class Depth(mmWave):
    def __init__(self, driver_path, train=True, **kwargs):
        enable_sources = ["arbe","master","kinect_pcl","kinect_color","mesh","mosh","mesh_param"]
        super().__init__(driver_path, train, num_points=4096, enable_sources=enable_sources, **kwargs)

    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = []
        seq_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        for d_path in map(str, self.driver_path.split(',')):
            seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]
        
        process_dict = {}
        self.full_data = {}
        for path in seq_paths:
            seq_pkl = os.path.join(path, 'pkl_data/{}_rgb.pkl'.format(self.input_data))
            try:
                if os.path.exists(seq_pkl) and self.use_pkl:
                    with open(seq_pkl, 'rb') as f:
                        seq_loader = pickle.load(f)[self.skip_head:-self.skip_tail-1]
                else:
                    # init result loader, reindex
                    seq_loader = ResultFileLoader(path, self.skip_head, self.skip_tail, self.enable_sources)
                    if self.create_pkl:
                        p = Process(target=self.write_pkl, args=(path, seq_pkl, seq_loader))
                        p.start()
                        process_dict.update({path: p})
            except Exception as e:
                print(e)
                continue

            if len(seq_loader) < 6:
                continue
            self.full_data.update({path:seq_loader})
            self.seq_paths.append(path)
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
        if self.create_pkl:
            for path, p in process_dict.items():
                p.join()
                with open(os.path.join(path, 'pkl_data/{}_rgb.pkl'.format(self.input_data)), 'rb') as f:
                    self.full_data[path] = pickle.load(f)

    def load_data(self, seq_loader, idx):
        if isinstance(seq_loader, list):
            if self.feature_type == 'rgb':
                return seq_loader[idx]
            else:
                data, label = seq_loader[idx]
                if data is not None:
                    return data[:, :3], label
                else:
                    return data, label
        
        frame, _ = seq_loader[idx]
        kinect_pcl = frame['master_pcl']
        kinect_color = frame["master_color"]
        kinect_color = kinect_color.reshape(len(kinect_pcl), 3) / [255, 255, 255]

        # param: pose, shape
        if frame['mesh_param'] is None:
            return None, None
        
        mesh_pose = frame['mesh_param']['pose']
        mesh_shape = frame['mesh_param']['shape']
        mesh_joint = frame['mesh_param']['joints']

        # filter radar_pcl with bounding box
        kinect_data = pcl_filter(mesh_joint, np.hstack((kinect_pcl, kinect_color)), 0.2, 0.21)

        # remove bad frame
        if kinect_data.shape[0] == 0:
            return None, None
        
        bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0))/2)[:3]
        kinect_data[:,:3] -= bbox_center
        mesh_pose[:3] -= bbox_center

        # padding
        kinect_data = self.pad_data(kinect_data)
        label = np.concatenate((mesh_pose, mesh_shape), axis=0)

        if self.feature_type == 'rgb':
            return kinect_data, label
        else:
            return kinect_data[:, :3], label

from mosh.utils import mosh_pose_transform
import cv2
class mmFusion(mmWave):
    def __init__(self, driver_path, train=True, **kwargs):
        enable_sources = ['arbe','arbe_feature','master','kinect_pcl','kinect_color','mesh','mosh','mesh_param']
        # create_pkl = False
        create_pkl = True
        if 'create_pkl' not in kwargs.keys():
            kwargs.update({'create_pkl': create_pkl})
        super().__init__(driver_path, train, enable_sources=enable_sources, **kwargs)

    def load_data(self, seq_loader, idx):
        if isinstance(seq_loader, list):
            return seq_loader[idx]
        
        frame, _ = seq_loader[idx]
        arbe_pcl = frame['arbe']

        # param: pose, shape
        if frame['mesh_param'] is None:
            return None, None
        
        mesh_pose = frame['mesh_param']['pose']
        mesh_shape = frame['mesh_param']['shape']
        mesh_joint = frame['mesh_param']['joints']
        image = frame['master_color']
        trans_mat = seq_loader.trans['kinect_master']

        # filter radar_pcl with bounding box
        arbe_data = pcl_filter(mesh_joint, arbe_pcl, 0.2)

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None

        data = {}
        if self.feature_type == 'image_feature':
            trans_joint = (mesh_joint - trans_mat['t']) @ trans_mat['R']
            crop_image = image_crop(trans_joint, image)[0]
            image = cv2.resize(crop_image/255, (224, 224))

            bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0))/2)[:3]
            arbe_data -= bbox_center
            mesh_pose[:3] -= bbox_center
            
        elif self.feature_type == 'feature_map':
            # transform pcl coordinate to kinect master
            trans_pcl = (arbe_data - trans_mat['t']) @ trans_mat['R']
            trans_joint = (mesh_joint - trans_mat['t']) @ trans_mat['R']
            trans, root_orient = mosh_pose_transform(mesh_pose[:3], mesh_pose[3:6], mesh_joint[0], trans_mat)
            mesh_pose = np.hstack((trans, root_orient.reshape(-1), mesh_pose[6:]))
            crop_image, img_left_top, img_right_bottom = image_crop(trans_joint, image)
            img_center = (img_right_bottom - img_left_top)/2
            trans_pcl = self.pad_data(trans_pcl)
            pcl_2d = pcl_project(trans_pcl)
            pcl_2d = (pcl_2d -  img_left_top - img_center)/img_center
            image = cv2.resize(crop_image/255, (224, 224))

            bbox_center = ((trans_joint.max(axis=0) + trans_joint.min(axis=0))/2)[:3]
            arbe_data = trans_pcl - bbox_center
            mesh_pose[:3] -= bbox_center
            data.update({'pcl_2d': pcl_2d})

        # padding
        arbe_data = self.pad_data(arbe_data)
        label = np.concatenate((mesh_pose, mesh_shape), axis=0)
        data.update({'pcl':arbe_data, 'img':image})

        return data, label

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        pcl_clip = []
        img_clip = []
        pcl_2d = []
        seq_loader =self.full_data[seq_path]
        data, label = self.load_data(seq_loader, frame_idx)

        while data is None:
            frame_idx = random.randint(self.clip_range-1, len(seq_loader)-1)
            data, label = self.load_data(seq_loader, frame_idx)

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.clip_step):
            # get xyz and features
            clip_data, _ = self.load_data(seq_loader, clip_id)
            # remove bad frame
            if clip_data is None:
                clip_data = data
            # padding
            pcl_clip.append(clip_data['pcl'])
            img_clip.append(clip_data['img'])
            if self.feature_type == 'feature_map':
                pcl_2d.append(clip_data['pcl_2d'])

        input_data = {}
        pcl_clip.append(data['pcl'])
        img_clip.append(data['img'])
        if self.feature_type == 'feature_map':
            pcl_2d.append(clip_data['pcl_2d'])
            pcl_2d = np.asarray(pcl_2d, dtype=np.float32)
            input_data.update({'pcl_2d':pcl_2d})
        pcl_clip = np.asarray(pcl_clip, dtype=np.float32)
        img_clip = np.asarray(img_clip, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)
        if True in np.isnan(label):
            label = np.nan_to_num(label)
        input_data.update({'pcl':pcl_clip, 'img':img_clip})

        return input_data, label, (seq_idx, frame_idx)


from mosh.utils import mosh_pose_transform
import cv2
class DepthFusion(mmFusion):
    def __init__(self, driver_path, train=True, **kwargs):
        super().__init__(driver_path, train, num_points=4096, **kwargs)
        
    def load_data(self, seq_loader, idx):
        if isinstance(seq_loader, list):
            return seq_loader[idx]

        frame, _ = seq_loader[idx]
        kinect_pcl = frame['master_pcl']

        # param: pose, shape
        if frame['mesh_param'] is None:
            return None, None
        
        mesh_pose = frame['mesh_param']['pose']
        mesh_shape = frame['mesh_param']['shape']
        mesh_joint = frame['mesh_param']['joints']
        image = frame["master_color"]
        trans_mat = seq_loader.trans['kinect_master']
        
        # filter radar_pcl with bounding box
        kinect_data = pcl_filter(mesh_joint, kinect_pcl, 0.2, 0.21)

        # remove bad frame
        if kinect_data.shape[0] == 0:
            return None, None
                
        if self.feature_type == 'image_feature':
            trans_joint = (mesh_joint - trans_mat['t']) @ trans_mat['R']
            crop_image = image_crop(trans_joint, image)[0]
            image = cv2.resize(crop_image/255, (224, 224))

            bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0))/2)[:3]
            kinect_data -= bbox_center
            mesh_pose[:3] -= bbox_center

        elif self.feature_type == 'feature_map':
            # transform pcl coordinate to kinect master
            trans_pcl = (kinect_data - trans_mat['t']) @ trans_mat['R']
            trans_joint = (mesh_joint - trans_mat['t']) @ trans_mat['R']
            trans, root_orient = mosh_pose_transform(mesh_pose[:3], mesh_pose[3:6], mesh_joint[0], trans_mat)
            mesh_pose = np.hstack((trans, root_orient.reshape(-1), mesh_pose[6:]))
            crop_image = image_crop(trans_joint, image)[0]
            image = cv2.resize(crop_image/255, (224, 224))

            bbox_center = ((trans_joint.max(axis=0) + trans_joint.min(axis=0))/2)[:3]
            kinect_data = trans_pcl - bbox_center
            mesh_pose[:3] -= bbox_center

        # padding
        kinect_data = self.pad_data(kinect_data)
        label = np.concatenate((mesh_pose, mesh_shape), axis=0)

        return {'pcl':kinect_data, 'img':image}, label


class MMMoshPKL(mmWave):
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = []
        seq_paths = []
        self.selected_dirs = TRAIN_DIRS if self.train else SELECTED_DIRS[self.test_data]
        for d_path in map(str, self.driver_path.split(',')):
            seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in self.selected_dirs]

        # init result loader, reindex
        for path in seq_paths:
            pkl_fname = os.path.join(path, 'pkl_data/data.pkl')
            try:
                with open(pkl_fname, 'rb') as f:
                    seq_data = pickle.load(f, encoding='bytes')
            except Exception as e:
                print(e)
                continue
            seq_len = len(seq_data) - self.skip_head
            if seq_len > 6:
                self.seq_paths.append(pkl_fname)
                self.index_map.append(self.index_map[-1] + seq_len)

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        frame_idx += self.skip_head
        seq_fname = self.seq_paths[seq_idx]
        with open(seq_fname, 'rb') as f:
            seq = pickle.load(f, encoding='bytes')
        
        clip = []
        data = seq[frame_idx]

        for clip_id in range(frame_idx-self.clip_range+1, frame_idx, self.clip_step):
            # get xyz and features
            clip_data = seq[clip_id]
            # clip padding
            clip.append(clip_data[self.data_device])
        clip.append(data[self.data_device])
        clip = np.asarray(clip, dtype=np.float32)
        label = np.asarray(data['mesh_param'], dtype=np.float32)
        if True in np.isnan(clip):
            label = np.nan_to_num(clip)
        if True in np.isnan(label):
            label = np.nan_to_num(label)
            
        return clip, label, (seq_idx, frame_idx)

