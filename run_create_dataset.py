import time
import os
import cv2
import numpy as np
from dataloader.result_loader import ResultFileLoader
import zipfile

def zip_files(source_path: str, target_path: str):
    zip = zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(source_path):
        fpath = path.replace(source_path, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

def make_dirs(save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, 'radar'))
        os.makedirs(os.path.join(save_path, 'image'))
        os.makedirs(os.path.join(save_path, 'joints'))

def save_data(save_path: str, seq_loader: ResultFileLoader):
    trans_mat = seq_loader.trans['kinect_master'].tolist()
    with open(os.path.join(save_path, 'calib.txt'), 'w') as f:
        f.write("{}".format(trans_mat))
    for idx in range(len(seq_loader)):
        frame, _ = seq_loader[idx]
        arbe_pcl = frame['arbe']
        arbe_feature = frame['arbe_feature'][:, [0,4,5]]
        rgb_img = frame['master_color']
        mesh_joint = frame['mesh_param']['joints'][:22]
        arbe_data = np.hstack((arbe_pcl, arbe_feature))
        np.save(os.path.join(save_path, 'radar', 'frame_id={}'.format(idx)), arbe_data)
        cv2.imwrite(os.path.join(save_path, 'image', 'frame_id={}.png'.format(idx)), rgb_img)
        np.save(os.path.join(save_path, 'joints', 'frame_id={}'.format(idx)), mesh_joint)

def main():
    driver_path = '/home/nesc525/drivers/1,/home/nesc525/drivers/2,/home/nesc525/drivers/3'
    train_paths = []
    test_paths = {}
    
    train_dirs = ['2021-10-18_18-36-00_M','2021-10-18_18-45-27_M',
                '2021-10-19_09-21-56_M','2021-10-19_09-50-21_M',
                '2021-10-19_14-41-26_F','2021-10-19_14-44-01_F',
                '2021-10-20_14-08-40_F','2021-10-20_14-36-51_F',
                '2021-10-20_19-59-21_F','2021-10-20_20-06-52_F',
                '2022-03-25_14-38-13_M','2022-03-25_14-41-05_M','2022-03-25_14-44-24_M','2022-03-25_14-46-35_M','2022-03-25_14-53-27_M',
                '2022-03-29_11-39-34_F','2022-03-29_11-42-36_F','2022-03-29_11-47-43_F','2022-03-29_11-50-15_F','2022-03-29_11-52-26_F',]
    test_dirs = dict(
        lab1 = '2022-03-25_16-54-28_M',
        lab2 = '2021-10-17_14-49-59_F',
        furnished = '2022-03-30_21-57-02_F',
        rain = '2022-03-30_20-49-18_M',
        smoke = '2022-03-25_17-10-08_M',
        poor_lighting = '2022-03-30_22-11-03_F',
    )
    for d_path in map(str, driver_path.split(',')):
        train_paths += [os.path.join(d_path, p) for p in train_dirs if p in os.listdir(d_path)]
        test_paths.update({k:os.path.join(d_path, v) for k, v in test_dirs.items() if v in os.listdir(d_path)})

    enable_sources = ["arbe","arbe_feature","master","kinect_color","mesh","mosh","mesh_param"]

    for i, seq_path in enumerate(train_paths):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        seq_loader = ResultFileLoader(seq_path, 0, 0, enable_sources)
        save_path = os.path.join('/home/nesc525/drivers/0/chen/dataset/train', 'subject{}'.format(i))
        make_dirs(save_path)
        save_data(save_path, seq_loader)

    for k, v in test_paths.items():
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        seq_loader = ResultFileLoader(v, 0, 0, enable_sources)
        save_path = os.path.join('/home/nesc525/drivers/0/chen/dataset/test', '{}'.format(k))
        make_dirs(save_path)
        save_data(save_path, seq_loader)
        
    source_path = '/home/nesc525/drivers/0/chen/dataset'
    target_path = "/home/nesc525/drivers/0/chen/dataset.zip"
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    zip_files(source_path, target_path)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print('Done')

if __name__ == "__main__":
    main()