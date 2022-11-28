import numpy as np
import cv2
import os
from pyk4a import CalibrationType, PyK4APlayback
from calib.utils import kinect_transform_mat, optitrack_transform_mat
from sync.offsets import Offsets
from dataloader.result_loader import ResultFileLoader
from nn.datasets.folder_list import TRAIN_DIRS, TEST_DIRS
import joblib
data_paths = ['/home/nesc525/drivers/2/','/home/nesc525/drivers/1/','/home/nesc525/drivers/3/']
choice_dir = TRAIN_DIRS
VIBE_DB_DIR = '/home/nesc525/chen/VIBE/data/vibe_db'
db = {
    'theta': [],
    'vid_name': [],
}
theta_len = 82
thetas = np.zeros([theta_len])
vid_names = ['2021-10-19_14-41-26_T']
for data_path in data_paths:
    for vid_name in choice_dir:
        dir_path = os.path.join(data_path, vid_name,'mosh','param')
        if not os.path.exists(dir_path):
            continue
        print(data_path+vid_name)
        npz_list = os.listdir(dir_path)
        ii = 0
        for i in range(len(npz_list)):
                if npz_list[i-ii].split('.')[-1] != 'npz':        
                    del npz_list[i-ii]
                    ii+=1 
        npz_list.sort(key = lambda x: int(x.split('_')[0][3:]))

        for i, npz in enumerate(npz_list):
            data = np.load(os.path.join(dir_path,npz))
            pose = data['pose'][3:]
            shape = data['shape']
            theta = np.concatenate([pose,shape], axis=0)
            thetas=np.hstack((thetas,theta))
            vid_names=np.hstack((vid_names,vid_name))
db['vid_name'] = vid_names[1:]
db['theta'] = thetas.reshape([int(len(thetas)/theta_len),theta_len])[1:]
db_file = os.path.join(VIBE_DB_DIR, 'our_smplx_db.pt')
print(f'Saving smplx dataset to {db_file}')
joblib.dump(db, db_file)



    