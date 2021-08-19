import numpy as np
import open3d as o3d
from itertools import compress
from dataloader.result_loader import KinectResultLoader, ArbeResultLoader
import os
import time

class SkelArbeManager():
    def __init__(self, result_path, *devices) -> None:
        if not devices:
            devices = ("master","sub1","sub2")
        self.devices = devices
        self.k_loader_dict = {}
        for device in devices:
            param = [dict(tag="kinect/{}/skeleton".format(device), ext=".npy")]
            self.k_loader_dict.update({device:KinectResultLoader(result_path, param)})
        self.a_loader = ArbeResultLoader(result_path)

    def generator(self, device="master"):
        if device not in self.devices:
            print("No {}".format(device))
            exit(1)
        param = "kinect/{}/skeleton".format(device)
        for i in range(len(self.a_loader)):
            a_row = self.a_loader[i]
            k_row = self.k_loader_dict[device].select_item(a_row["arbe"]["tm"], "st", False)
            # k_row = self.k_loader_dict[device].select_by_skid(i)
            # a_row = self.a_loader.select_item(k_row[param]["st"], "tm", False)
            yield k_row[param], a_row["arbe"]

def pcd_visualization(filepath, *devices) -> None:
    """
        load kinect skeleton and arbe point_cloud to visualize.
    """
    sam = SkelArbeManager(filepath, *devices)
    #创建可视化类vis
    vis = o3d.visualization.Visualizer()
    #添加窗口
    vis.create_window(width=800, height=1200)

    #创建骨骼，点云，线条类
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    line_set = o3d.geometry.LineSet()
    o3d_arbe_pcd = o3d.geometry.PointCloud()
    o3d_skel_pcd = o3d.geometry.PointCloud()
    #创建旋转矩阵R
    # R = axis_pcd.get_rotation_matrix_from_xyz((np.pi,np.pi*3/4,0))
    # axis_pcd.rotate(R,center=[0,0,0])
    to_reset = True
    #对于图像进行放缩
    # o3d_skel_pcd.scale(0.1,center=o3d_skel_pcd.get_center())
    # o3d_arbe_pcd.scale(0.1,center=o3d_arbe_pcd.get_center())
    # axis_pcd.scale(0.5,center=axis_pcd.get_center())
    # line_set.scale(0.1,center=o3d_skel_pcd.get_center())
    #对于图像进行显示
    vis.add_geometry(axis_pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(o3d_arbe_pcd)
    vis.add_geometry(o3d_skel_pcd)

    
    for s_row, a_row in sam.generator("master"):
        #读取相关npy文件
        s_np = np.load(s_row["filepath"])
        a_np = np.load(a_row["filepath"])
        person_count = s_np.shape[0]
        kinect_skeleton = s_np[:,:,:3].reshape(-1,3)/1000
        #坐标系变换
        rotation = np.array([[1,0,0],
                            [0,0,-1],
                            [0,1,0]])
        translation = np.array([0, -0.05, 0.2]).T
        skeleton_pcd = kinect_skeleton.dot(rotation) + translation
        #对于点云进行切割
        k_max = skeleton_pcd.max(axis=0) + 0.5
        k_min = skeleton_pcd.min(axis=0) - 0.5
        all_arbe_pcd = a_np[:,:3]
        a_in_k = (all_arbe_pcd < k_max) & (all_arbe_pcd > k_min)

        filter_list = []
        for row in a_in_k:
            filter_list.append(False if False in row else True)
        arbe_pcd = np.array(list(compress(all_arbe_pcd, filter_list)))
        #绘制线条
        lines_skel = np.array([[0,1],[1,2],[2,3],[2,4],[4,5],[5,6],[6,7],[7,8],
                            [8,9],[7,10],[2,11],[11,12],[12,13],[13,14],[14,15],
                            [15,16],[14,17],[0,18],[18,19],[19,20],[20,21],[0,22],
                            [22,23],[23,24],[24,25],[3,26],[26,27],[26,28],[26,29],
                            [26,30],[26,31]])
        if person_count > 1:
            for p in range(1, person_count):
                lines_skel = np.vstack((lines_skel, lines_skel+p*31+1))
        lines_colors = np.array([[0, 0, 1] for j in range(len(lines_skel))])
        #更新点云，骨骼点
        o3d_skel_pcd.points = o3d.utility.Vector3dVector(skeleton_pcd)
        o3d_skel_pcd.paint_uniform_color([0,0,1])
        o3d_arbe_pcd.points = o3d.utility.Vector3dVector(arbe_pcd)
        o3d_arbe_pcd.paint_uniform_color([0,1,0])
        #对图像进行旋转
        # R1 = axis_pcd.get_rotation_matrix_from_xyz((-np.pi/2,0,np.pi))
        # R2 = axis_pcd.get_rotation_matrix_from_xyz((-np.pi/2,np.pi/2,np.pi/2))
        # o3d_skel_pcd.rotate(R1,center=o3d_skel_pcd.get_center())
        # o3d_skel_pcd.rotate(R2,center=(0,0,0))
        # o3d_skel_pcd.scale(0.5,center =o3d_skel_pcd.get_center())
        # o3d_arbe_pcd.rotate(R1,center=o3d_arbe_pcd.get_center())
        # o3d_arbe_pcd.rotate(R2,center=(0,0,0))
        # o3d_arbe_pcd.scale(0.5,center=o3d_arbe_pcd.get_center())
        # line_set.scale(0.5,center=o3d_skel_pcd.get_center())
        #o3d_arbe_pcd = o3d.geometry.PointCloud()
        #更新线条点
        line_set.points = o3d_skel_pcd.points
        line_set.lines = o3d.utility.Vector2iVector(lines_skel)
        line_set.colors = o3d.utility.Vector3dVector(lines_colors)
        #更新图像
        vis.update_geometry(line_set)
        vis.update_geometry(o3d_arbe_pcd)
        vis.update_geometry(o3d_skel_pcd)
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.02)
        # ctr = o3d.visualization.ViewControl()
        # ctr.change_field_of_view(step = -10)
        # print(axis_pcd.points)
        # o3d.visualization.draw_geometries([o3d_skel_pcd]+[o3d_arbe_pcd]+[axis_pcd]+[line_set])
        # o3d.io.write_point_cloud(os.path.join(filepath, "{}.ply".format(s_row["id"])), o3d_pcd)


def vis_skel_pcl(skel=None, jnts=None, pcl=None):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    if skel is not None:
        o3d_skel_pcd = o3d.geometry.PointCloud()
        o3d_skel_pcd.points = o3d.utility.Vector3dVector(skel.reshape(-1,3))
        o3d_skel_pcd.paint_uniform_color([0,0,1])
    if jnts is not None:
        o3d_jnts_pcd = o3d.geometry.PointCloud()
        o3d_jnts_pcd.points = o3d.utility.Vector3dVector(jnts.reshape(-1,3))
        o3d_jnts_pcd.paint_uniform_color([0,1,0])
    if pcl is not None:
        o3d_pcls_pcd = o3d.geometry.PointCloud()
        o3d_pcls_pcd.points = o3d.utility.Vector3dVector(pcl.reshape(-1,3))
        o3d_pcls_pcd.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([axis_pcd, o3d_pcls_pcd, o3d_skel_pcd, o3d_jnts_pcd], 'skel-pcl')


if __name__ == "__main__":
    k_np = np.load("__test__/2021-08-05 17:21:35/kinect/master/skeleton/id=122_tm=21161244_st=1628155321.3146186.npy")
    a_np = np.load("__test__/2021-08-05 17:21:35/arbe/id=317_ts=1628155321.305585.npy")
    # pcd_visualization("", k_np, a_np)

