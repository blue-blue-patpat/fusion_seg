import os
import numpy as np
from dataloader.utils import extract_transform_mat
from kinect.config import EXTRINSIC_MAS_ARBE


def kinect_transform_mat(root_path: str):
    input_path = os.path.join(root_path, "calib/kinect")
    R_mas_to_world, t_mas_to_world = extract_transform_mat(os.path.join(input_path, "master_to_world.npz"))
    R_sub1_to_world, t_sub1_to_world = extract_transform_mat(os.path.join(input_path, "sub1_to_world.npz"))
    R_sub2_to_world, t_sub2_to_world = extract_transform_mat(os.path.join(input_path, "sub2_to_world.npz"))

    R_mas_to_radar = EXTRINSIC_MAS_ARBE["R"]
    t_mas_to_radar = EXTRINSIC_MAS_ARBE["t"]
    
    R_sub1_to_mas = np.linalg.inv(R_mas_to_world) @ R_sub1_to_world
    t_sub1_to_mas = np.linalg.inv(R_mas_to_world) @ (t_sub1_to_world - t_mas_to_world)

    R_sub2_to_mas = np.linalg.inv(R_mas_to_world) @ R_sub2_to_world
    t_sub2_to_mas = np.linalg.inv(R_mas_to_world) @ (t_sub2_to_world - t_mas_to_world)

    return dict(
        kinect_master=dict(
            R=R_mas_to_radar,
            t=t_mas_to_radar
        ),
        kinect_sub1=dict(
            R=R_mas_to_radar @ R_sub1_to_mas,
            t=R_mas_to_radar @ t_sub1_to_mas + t_mas_to_radar
        ),
        kinect_sub2=dict(
            R=R_mas_to_radar @ R_sub2_to_mas,
            t=R_mas_to_radar @ t_sub2_to_mas + t_mas_to_radar
        )
    )


def optitrack_transform_mat(root_path: str):
    input_path = os.path.join(root_path, "calib/optitrack")
    R_opti_to_radar, t_opti_to_radar = extract_transform_mat(os.path.join(input_path, "optitrack_to_radar.npz"))
    return dict(
        optitrack=dict(
            R=R_opti_to_radar,
            t=t_opti_to_radar
        )
    )


def to_radar_transform_mat(root_path: str):
    """
    Compute device coordinate to radar coordinate transform matrixes

    :return:
    |--kinect_master--{R, t}
    |--kinect_sub1  --{R, t}
    |--kinect_sub2  --{R, t}
    |--optitrack    --{R, t}
    """
    transform_dict = {}
    try:
        transform_dict.update(kinect_transform_mat(root_path))
    except:
        print("Kinect Rt not found.")
    try:
        transform_dict.update(optitrack_transform_mat(root_path))
    except:
        print("OptiTrack Rt not found.")
    return transform_dict


def kinect_to_world_transform_cpp(root_path, devices:list=["master","sub1","sub2"]):
    import json
    trans_mat = {}
    for dev in devices:
        R, t = np.load(root_path+"/calib/kinect/{}_to_world.npz".format(dev)).values()
        trans_mat[dev] = np.vstack((np.hstack((R, t.reshape(-1,1))), [0,0,0,1]))
    return trans_mat


def to_radar_rectified_trans_mat(root_path):
    offset_fname = os.path.join(root_path, 'calib_offsets.txt')
    if os.path.exists(offset_fname):
        with open(offset_fname, 'r') as f:
            calib_offset = eval(f.readline())
        trans_mat = to_radar_transform_mat(root_path)
        for k, v in calib_offset.items():
            if k in trans_mat.keys():
                trans_mat[k]['t'] = v
        return trans_mat