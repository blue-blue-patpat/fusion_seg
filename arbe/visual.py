import open3d as o3d
import numpy as np

class pt_Vis():
    def __init__(self,name='20m test',width=800,height=600,json='./config/view_point.json'):
        self.vis=o3d.visualization.Visualizer()
        self.vis.create_window(window_name=name,width=width,height=height)
        self.axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1
        opt.show_coordinate_frame = True


        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.param=o3d.io.read_pinhole_camera_parameters(json)
        self.ctr=self.vis.get_view_control()
        self.ctr.convert_from_pinhole_camera_parameters(self.param)
        print('viewpoint json loaded!')






    def __del__(self):
        self.vis.destroy_window()

    def update(self,pcd):
        '''

        :param pcd: PointCLoud()
        :return:
        '''
        self.pcd.points=pcd.points
        self.pcd=pcd

        # self.pcd.colors=pcd.colors

        # self.vis.clear_geometries()
        self.vis.update_geometry(self.pcd)          # 更新点云

        # self.vis.remove_geometry(self.pcd)          # 删除vis中的点云
        self.vis.add_geometry(self.pcd)             # 增加vis中的点云

        # 设置viewpoint
        self.ctr.convert_from_pinhole_camera_parameters(self.param)

        self.vis.poll_events()
        self.vis.update_renderer()
        # self.vis.run()

    def capture_screen(self,fn, depth = False):
        if depth:
            self.vis.capture_depth_image(fn, False)
        else:
            self.vis.capture_screen_image(fn, False)

if __name__ == '__main':
    pcd = pt_Vis()
    pcd.update('/home/nesc525/chen/3DSVC/arbe/cloud_1.ply')

