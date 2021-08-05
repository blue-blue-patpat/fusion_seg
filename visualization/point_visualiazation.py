import numpy as np
import open3d as o3d
import os
import time 
from pyntcloud import PyntCloud
from functools import reduce

class Npy2Point:
    def __init__(self,filepath):
        from dataloader.result_loader import ResultManager
        rm = ResultManager(filepath)
        for k_np, a_np in rm.generator():
            self.run(k_np, a_np)
    
    def run(self, k_np, a_np):
        for i in range(self.count):
            arbe_points = a_np[:,:3].reshape(-1,3)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dvector()
            o3d.visualization.draw_geometries([point_cloud])
            time.sleep(1)


points1 = np.load("__test__/2021-08-05 17:21:35/arbe/id=317_ts=1628155321.305585.npy")
points1 = points1.reshape(-1,3)
points2 = np.load("__test__/2021-08-05 17:21:35/kinect/master/skeleton/id=122_tm=21161244_st=1628155321.3146186.npy")
print(points2.shape)
'''
row_rand_array = np.arange(points.shape[0])
np.random.shuffle(row_rand_array)
points = points[row_rand_array[0:200]]
'''

points2 = np.vstack((points2[0],points2[1]))
points2 /= 1000
points2 = points2[:,:3]
#旋转变换
rotation = np.zeros((3,3))
rotation[0][0] = 1
rotation[1][2] = -1
rotation[2][1] = 1
points2 = np.dot(points2,rotation)
#平移变换
#根据arbe的官方文档，arbe的坐标系原点位于top以下55.5mm，right以左71.5mm
points2[:,1] += 0.04
points2[:,2] += 0.08
xmin = np.min(points2[:,0])
xmax = np.max(points2[:,0])
ymin = np.min(points2[:,1])
ymax = np.max(points2[:,1])
zmin = np.min(points2[:,2])
zmax = np.max(points2[:,2])
index1 = np.intersect1d(np.where(points1[:,0]>=xmin-1),np.where(points1[:,0]<=xmax+1))
index2 = np.intersect1d(np.where(points1[:,1]>=ymin-1),np.where(points1[:,1]<=ymax+1))
index3 = np.intersect1d(np.where(points1[:,2]>=zmin-1),np.where(points1[:,2]<=zmax+1))
index = reduce(np.intersect1d,[index1,index2,index3])
points1 = points1[index]
points = np.vstack((points1,points2))
print(points2.shape)
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud]+[axis_pcd])