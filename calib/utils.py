import os
import numpy as np
from dataloader.utils import extract_transform_mat
from kinect.config import EXTRINSIC_MAS_ARBE


def kinect_transform_mat(root_path: str):
    input_path = os.path.join(root_path, "calib/kinect")
    R_mas_to_world, t_mas_to_world = extract_transform_mat(os.path.join(input_path, "master.npz"))
    R_sub1_to_world, t_sub1_to_world = extract_transform_mat(os.path.join(input_path, "sub1.npz"))
    R_sub2_to_world, t_sub2_to_world = extract_transform_mat(os.path.join(input_path, "sub2.npz"))

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
    R_opti_to_radar, t_opti_to_radar = extract_transform_mat(os.path.join(input_path, "transform.npz"))
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
    transform_dict.update(kinect_transform_mat(root_path))
    transform_dict.update(optitrack_transform_mat(root_path))
    return transform_dict
