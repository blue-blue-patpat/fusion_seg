import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d  # noqa: F401
import open3d as o3d

import pyk4a
from pyk4a import Config, PyK4A


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            camera_fps=pyk4a.FPS.FPS_5,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()
    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    geometry_added = False
    vis = o3d.visualization.Visualizer()
    vis.create_window("PointCloud")
    pcd = o3d.geometry.PointCloud()


    while True:
        capture = k4a.get_capture()

        points = capture.depth_point_cloud
        pointcloudcolor_np = capture.transformed_color[..., (2, 1, 0)]

        xyz = np.zeros((points.shape[0]*points.shape[1], 3), dtype=np.float32)
        x = points[:, :, 0]  # extract the X, Y, Z matrices
        y = points[:, :, 1]
        z = points[:, :, 2]
        xyz[:, 0] = np.reshape(x, -1) # convert to vectors
        xyz[:, 1] = np.reshape(y, -1)
        xyz[:, 2] = np.reshape(z, -1)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # Prepare the Pointcloud colors for Open3D pointcloud colors
        colors = np.zeros((pointcloudcolor_np.shape[0]*pointcloudcolor_np.shape[1], 3), dtype=np.float32)
        blue = pointcloudcolor_np[:, :, 0] # extract the blue, green , and red components
        green = pointcloudcolor_np[:, :, 1]
        red = pointcloudcolor_np[:, :, 2]
        colors[:, 0] = np.reshape(red, -1) / 255.0 # convert to vectors and scale to 0-1
        colors[:, 1] = np.reshape(green, -1) / 255.0
        colors[:, 2] = np.reshape(blue, -1) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if not geometry_added:
            vis.add_geometry(pcd)
            geometry_added = True
        
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()


    k4a.stop()


if __name__ == "__main__":
    main()
