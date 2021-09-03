import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

from dataloader.result_loader import KinectResultLoader, OptitrackResultLoader, ArbeResultLoader


def sync_kinect(root_path):
    mas_loader = KinectResultLoader(root_path)
    sub1_loader = KinectResultLoader(root_path)
    sub2_loader = KinectResultLoader(root_path)


x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(x, y, dist=euclidean)
distance, path