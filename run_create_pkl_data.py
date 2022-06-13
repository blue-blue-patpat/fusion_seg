import os
import time
import numpy as np
import pickle
from multiprocessing import Pool, Process
from nn.p4t.datasets.folder_list import *
from dataloader.result_loader import ResultFileLoader, PKLLoader

def write_pkl(pkl_path, seq_loader):
    if not os.path.exists(pkl_path):
        os.mkdir(pkl_path)
    for frame in seq_loader:
        t_s = time.time()
        arbe_fname = os.path.split(frame[1]['arbe']['filepath'])[1][:-4] + '.pkl'
        pkl_fname = os.path.join(pkl_path, arbe_fname)
        with open(pkl_fname, 'wb') as f:
            pickle.dump(frame, f)
        t_e = time.time()
        print(arbe_fname, t_e-t_s)
    print(pkl_path, 'Done')

if __name__ == "__main__":
    selected_dirs = []
    for v in SELECTED_DIRS.values():
        selected_dirs += v
    driver_path = '/home/nesc525/drivers/1,/home/nesc525/drivers/2,/home/nesc525/drivers/3'
    seq_paths = []
    for d_path in map(str, driver_path.split(",")):
        seq_paths += [os.path.join(d_path, p) for p in os.listdir(d_path) if p in selected_dirs]

    pool = Pool(40)
    pro_list = []
    for path in seq_paths:
        pkl_path = os.path.join(path, 'pkl')
        seq_loader = ResultFileLoader(path, 0, 0, ["arbe","arbe_feature","master","kinect_pcl","kinect_color","optitrack","mesh","mosh","mesh_param"])
        if len(seq_loader) < 6:
            continue
        p = Process(target=write_pkl, args=(pkl_path, seq_loader))
        p.start()
        pro_list.append(p)
        # pool.apply_async(write_pkl, (pkl_path, seq_loader))
        # write_pkl(pkl_path, seq_loader)

    for p in pro_list:
        p.join()
    # pool.close()
    # pool.join()
    