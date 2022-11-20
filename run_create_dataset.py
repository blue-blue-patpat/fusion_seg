import time
import os
import cv2
import numpy as np
import zipfile
from dataloader.result_loader import ResultFileLoader
from visualization.utils import image_crop, pcl_filter, o3d_plot, o3d_pcl, o3d_mesh

def zip_files(source_path: str, target_path: str):
    zip = zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(source_path):
        fpath = path.replace(source_path, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

def make_dirs(save_path: str):
    if not os.path.exists(os.path.join(save_path, 'radar')):
        os.makedirs(os.path.join(save_path, 'radar'))
    if not os.path.exists(os.path.join(save_path, 'image', 'master')):
        os.makedirs(os.path.join(save_path, 'image', 'master'))
    if not os.path.exists(os.path.join(save_path, 'image', 'sub')):
        os.makedirs(os.path.join(save_path, 'image', 'sub'))
    if not os.path.exists(os.path.join(save_path, 'depth', 'master')):
        os.makedirs(os.path.join(save_path, 'depth', 'master'))
    if not os.path.exists(os.path.join(save_path, 'depth', 'sub')):
        os.makedirs(os.path.join(save_path, 'depth', 'sub'))
    if not os.path.exists(os.path.join(save_path, 'depth_pcl', 'sub')):
        os.makedirs(os.path.join(save_path, 'depth_pcl', 'sub'))
    if not os.path.exists(os.path.join(save_path, 'mesh')):
        os.makedirs(os.path.join(save_path, 'mesh'))

def save_data(save_path: str, seq_loader: ResultFileLoader):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    trans_mat = seq_loader.calib_offsets.copy()
    trans_mat.pop('kinect_sub1')
    opti = trans_mat.pop('optitrack')
    trans_mat['kinect_sub'] = trans_mat.pop('kinect_sub2')
    trans_mat['optitrack'] = opti
    with open(os.path.join(save_path, 'calib.txt'), 'w') as f:
        f.write("{}".format(trans_mat))
    
    for idx in range(len(seq_loader)):
        # if idx < 1788:
        #     continue
        frame, _ = seq_loader[idx]
        mesh_joint = frame['mesh_param']['joints']
        mesh = frame['mesh_param']
        del mesh['faces']
        arbe_pcl = frame['arbe']
        arbe_feature = frame['arbe_feature'][:, [0,4,5]]
        arbe_data = np.hstack((arbe_pcl, arbe_feature))
        
        master_img = frame['master_color']
        crop_img = image_crop(mesh_joint, master_img, trans_mat=trans_mat['kinect_master'], square=True)[0]
        master_pcl = frame['master_pcl']
        master_color = master_img.reshape(len(master_pcl), 3) / [255, 255, 255]
        master_pcl = pcl_filter(mesh_joint, np.hstack((master_pcl, master_color)), 0.2, 0.21)
        master_img = cv2.resize(crop_img, (224, 224))
        master_depth_img = frame['master_depth']
        
        sub2_img = frame['sub2_color']
        crop_img = image_crop(mesh_joint, sub2_img, trans_mat=trans_mat['kinect_sub'], margin=0.2, square=True, cam="000176712112")[0]
        sub2_pcl = frame['sub2_pcl']
        sub2_color = sub2_img.reshape(len(sub2_pcl), 3) / [255, 255, 255]
        sub2_pcl = pcl_filter(mesh_joint, np.hstack((sub2_pcl, sub2_color)), 0.2, 0.21)
        sub2_img = cv2.resize(crop_img, (224, 224))
        sub2_depth_img = frame['sub2_depth']
        
        o3d_plot([o3d_pcl(master_pcl[:,:3]-mesh_joint[0], color=[0,0,1]), o3d_pcl(mesh_joint-mesh_joint[0], color=[0,1,0])])
        
        np.save(os.path.join(save_path, 'radar', 'frame_{}'.format(idx)), arbe_data)
        cv2.imwrite(os.path.join(save_path, 'image', 'master', 'frame_{}.png'.format(idx)), master_img)
        np.save(os.path.join(save_path, 'depth_pcl', 'master', 'frame_{}'.format(idx)), master_pcl)
        cv2.imwrite(os.path.join(save_path, 'depth', 'master', 'frame_{}.png'.format(idx)), master_depth_img)
        cv2.imwrite(os.path.join(save_path, 'image', 'sub', 'frame_{}.png'.format(idx)), sub2_img)
        np.save(os.path.join(save_path, 'depth_pcl', 'sub', 'frame_{}'.format(idx)), sub2_pcl)
        cv2.imwrite(os.path.join(save_path, 'depth', 'sub', 'frame_{}.png'.format(idx)), sub2_depth_img)
        np.savez(os.path.join(save_path, 'mesh', 'frame_{}'.format(idx)), **mesh)
        
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

def main():
    driver_path = '/home/nesc525/drivers/1,/home/nesc525/drivers/2,/home/nesc525/drivers/3'
    train_paths = []
    test_paths = dict(
        lab1 = [],
        lab2 = [],
        furnished = [],
        rain = [],
        smoke = [],
        poor_lighting = [],
        occlusion = [],
        # poor_lighting1 = [],
    )

    train_dirs = ['2021-10-18_18-36-00_M','2021-10-18_18-45-27_M',
                '2021-10-19_09-21-56_M','2021-10-19_09-50-21_M',
                '2021-10-19_14-41-26_F','2021-10-19_14-44-01_F',
                '2021-10-20_14-08-40_F','2021-10-20_14-36-51_F',
                '2021-10-20_19-59-21_F','2021-10-20_20-06-52_F',
                '2022-03-25_14-38-13_M','2022-03-25_14-41-05_M','2022-03-25_14-44-24_M','2022-03-25_14-46-35_M','2022-03-25_14-53-27_M',
                '2022-03-29_11-39-34_F','2022-03-29_11-42-36_F','2022-03-29_11-47-43_F','2022-03-29_11-50-15_F','2022-03-29_11-52-26_F',]
    
    test_dirs = dict(
        lab1 = ['2022-03-25_16-54-28_M','2022-03-29_11-11-21_F',],
        lab2 = ['2021-10-17_14-49-59_F','2021-10-18_09-48-31_M',],
        furnished = ['2022-03-30_21-57-02_F','2022-03-30_22-35-03_M',],
        rain = ['2022-03-30_20-49-18_M','2022-03-30_21-17-14_F',],
        smoke = ['2022-03-25_17-10-08_M','2022-03-25_17-17-23_M',],
        poor_lighting = ['2022-03-30_22-11-03_F','2022-03-30_22-27-35_M',],
        occlusion = ['2021-10-23_20-40-32_F','2021-10-23_21-03-26_M',],
        # poor_lighting1 = ['2022-03-30_21-31-14_F','2022-03-30_21-33-39_F',],
    )
    
    for d_path in map(str, driver_path.split(',')):
        train_paths += [os.path.join(d_path, p) for p in train_dirs if p in os.listdir(d_path)]
        for k, v in test_dirs.items():
            for p in v:
                if p in os.listdir(d_path):
                    test_paths[k].append(os.path.join(d_path, p))

    enable_sources = ["arbe","arbe_feature",'master','sub2',"kinect_color","kinect_depth","kinect_pcl","mesh","mosh","mesh_param"]
    
    for i, seq_path in enumerate(train_paths):
        # if i < 10 or i > 14:
        #     continue
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        seq_loader = ResultFileLoader(seq_path, 0, 0, enable_sources)
        print("seq_path", seq_path)
        save_path = os.path.join('/home/nesc525/drivers/6/dataset/train', 'sequence_{}'.format(i))
        make_dirs(save_path)
        save_data(save_path, seq_loader)

    for k, v in test_paths.items():
        if k != 'rain':
            continue
        for i, seq_path in enumerate(v):
            # if i == 0:
            #     continue
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            seq_loader = ResultFileLoader(seq_path, 0, 0, enable_sources)
            save_path = os.path.join('/home/nesc525/drivers/6/dataset/test', '{}'.format(k), 'sequence_{}'.format(i))
            make_dirs(save_path)
            save_data(save_path, seq_loader)

if __name__ == "__main__":
    main()