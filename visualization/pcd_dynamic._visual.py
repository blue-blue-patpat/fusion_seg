import os
import numpy as np
import open3d as o3d
import time 

files = os.listdir("__test__/2021-08-06 14:09:37/kinect/master/skeleton/")
vis = o3d.visualization.Visualizer()
vis.create_window()
pointcloud = o3d.geometry.PointCloud()
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
R = axis_pcd.get_rotation_matrix_from_xyz((np.pi,np.pi*3/4,0))
axis_pcd.rotate(R,center=[0,0,0])
to_reset = True
pointcloud.scale(0.1,center=pointcloud.get_center())
axis_pcd.scale(0.5,center=axis_pcd.get_center())
vis.add_geometry(pointcloud)
vis.add_geometry(axis_pcd)
o3d.visualization.show_coordinate_frame = True
for f in files:
    pcd = np.load("__test__/2021-08-06 14:09:37/kinect/master/skeleton/" + f)   #此处读取npy文件
    pcd = pcd[:,:,:3].reshape(-1,3)/1000
    pointcloud.points = o3d.utility.Vector3dVector(pcd)  
    R2 = axis_pcd.get_rotation_matrix_from_xyz((0,np.pi*3/4,0))
    pointcloud.rotate(R2,center=(0,0,0))
    pointcloud.paint_uniform_color([0,0,1])
    pointcloud.scale(0.5,center=pointcloud.get_center())
    vis.update_geometry(pointcloud)
    if to_reset:
        vis.reset_view_point(True)
        to_reset = False
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
'''
pcd = np.load("__test__/2021-08-06 14:09:37/kinect/master/skeleton/id=0_tm=3261266_st=1628230258.4281578.npy")
pcd = pcd[:,:,:3].reshape(-1,3)/1000
pointcloud = o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(pcd)
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
R = axis_pcd.get_rotation_matrix_from_xyz((0,0,np.pi))
axis_pcd.rotate(R,center=[0,0,0])
pointcloud.rotate(R,center=[0,0,0])
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pointcloud)
vis.add_geometry(axis_pcd)
ctr = vis.get_view_control()
ctr.change_field_of_view(step = -10)
print(vis.get_render_option())
vis.run()
'''