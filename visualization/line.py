import open3d as o3d
import numpy as np

def custom_draw_geometry(pcd,linesets):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(linesets)
    render_option = vis.get_render_option()
    render_option.point_size = 4
    render_option.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
	
    points_box = ***   # 3D框的8个点
    pc = ***   # 3维点云

  
    lines_box = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],
                        [8,9],[7,10],[2,11],[11,12],[12,13],[13,14],[14,15],
                        [15,16],[14,17],[0,18],[18,19],[19,20],[20,21],[0,22],
                        [22,23],[23,24],[24,25],[3,26],[26,27],[26,28],[26,29],
                        [26,30],[26,31]])
    colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc[:,:3])
    custom_draw_geometry(point_cloud, line_set)
