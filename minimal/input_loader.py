import numpy as np
from dataloader.result_loader import KinectResultLoader, OptitrackResultLoader
from minimal.bridge import JointsBridge
from calib.utils import to_radar_transform_mat


def stream_k_jnts_k_pcls(root_path):
    loader = KinectResultLoader(
        root_path,
        params=[
            dict(tag="kinect/master/pcls", ext=".npy"),
            dict(tag="kinect/sub1/pcls", ext=".npy"),
            dict(tag="kinect/sub2/pcls", ext=".npy"),

            dict(tag="kinect/master/skeleton", ext=".npy"),
            dict(tag="kinect/sub1/skeleton", ext=".npy"),
            dict(tag="kinect/sub2/skeleton", ext=".npy"),
    ])
    
    # brg = JointsBridge()
    
    trans_mats = to_radar_transform_mat(root_path)

    for item in loader.generator_by_skeleton():
        pcl = np.vstack(
            np.load(item["kinect/master/pcls"]["filepath"]) @ trans_mats["kinect_master"]["R"].T + trans_mats["kinect_master"]["t"],
            np.load(item["kinect/sub1/pcls"]["filepath"]) @ trans_mats["kinect_sub1"]["R"].T + trans_mats["kinect_sub1"]["t"],
            np.load(item["kinect/sub2/pcls"]["filepath"]) @ trans_mats["kinect_sub2"]["R"].T + trans_mats["kinect_sub2"]["t"],
        )

        skeleton = np.load(item["kinect/master/skeleton"]["filepath"]) @ trans_mats["kinect_master"]["R"].T + trans_mats["kinect_master"]["t"]

        # only process skeleton 0
        yield skeleton, pcl


def stream_o_jnts_k_pcls(root_path, time_key = "st"):
    k_loader = KinectResultLoader(
        root_path,
        params=[
            dict(tag="kinect/master/pcls", ext=".npy"),
            dict(tag="kinect/sub1/pcls", ext=".npy"),
            dict(tag="kinect/sub2/pcls", ext=".npy"),

            dict(tag="kinect/master/skeleton", ext=".npy"),
            dict(tag="kinect/sub1/skeleton", ext=".npy"),
            dict(tag="kinect/sub2/skeleton", ext=".npy"),
    ])

    o_loader = OptitrackResultLoader(root_path)
    
    # brg = JointsBridge()
    
    trans_mats = to_radar_transform_mat(root_path)

    i = 0
    for k_frame in k_loader.generator_by_skeleton():
        frame_time = k_frame["kinect/master/pcls"][time_key]
        o_frame = o_loader.select_item(frame_time, 'st', exact=False)

        pcl = np.vstack(
            np.load(k_frame["kinect/master/pcls"]["filepath"]) @ trans_mats["kinect_master"]["R"].T + trans_mats["kinect_master"]["t"],
            np.load(k_frame["kinect/sub1/pcls"]["filepath"]) @ trans_mats["kinect_sub1"]["R"].T + trans_mats["kinect_sub1"]["t"],
            np.load(k_frame["kinect/sub2/pcls"]["filepath"]) @ trans_mats["kinect_sub2"]["R"].T + trans_mats["kinect_sub2"]["t"],
        )

        skeleton = np.load(o_frame["optitrack"]["filepath"]) @ trans_mats["optitrack"]["R"].T + trans_mats["optitrack"]["t"]

        # only process skeleton 0
        yield skeleton, pcl, "id={}_{}={}".format(i, time_key, frame_time)
