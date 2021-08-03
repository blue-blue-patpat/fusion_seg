import numpy as np
import open3d as o3d
import os
import time 
from pyntcloud import PyntCloud

class Npy2Cloud:
    def __init__(self,filepath):#初始化类的属性
        import numpy as np
        from dataloader.result_loader import ResultManager
        rm = ResultManager(filepath)
        for k_np, k_img, a_np in rm.generator():
            self.run(k_np, k_img, a_np)
    
    #定义内部函数,实现功能
    def run(self, k_np, k_img, a_np):
        for i in range(self.count):
            points = np.load("points.npy")
            points = points.reshape(-1,3)
            print(points.shape)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dvector(points)
            o3d.visualization.draw_geometries([point_cloud])
            time.sleep(1)


points = np.load("__test__/2021-07-31 21:35:50/arbe/id=48_ts=1627738560.377321.npy")
points = points.reshape(-1,3)
'''
row_rand_array = np.arange(points.shape[0])
np.random.shuffle(row_rand_array)
points = points[row_rand_array[0:200]]
'''
points /= 1000

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud])