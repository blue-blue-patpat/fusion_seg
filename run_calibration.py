import os
import cv2
import numpy as np
import open3d as o3d
from kinect.calib import calibrate_kinect
from kinect.config import * 
from dataloader.utils import clean_dir
from dataloader.result_loader import KinectResultLoader
from visualization.utils import o3d_plot, o3d_pcl, o3d_coord
import argparse


def compute_single_transform(color_frame: np.ndarray, pcl_frame: np.ndarray, intrinsic: np.ndarray, aruco_size: float):
    """
    Compute R, t and colored pcl for a single camera

    :param color_frame: w*h*3 rgb array
    :param pcl_frame: w*h*3 transformed pcl array
    :param intrinsic: 3*3 array
    :param aruco_size: tag size in mm
    :return: 3*3 R, 1*3 t, open3d.PointCloud pcl
    """
    # R, t is rotation and translate matrix from world to camera
    R, t = calibrate_kinect(color_frame, pcl_frame, intrinsic_mtx=intrinsic, aruco_size=aruco_size)
    pcl = o3d_pcl((np.linalg.inv(R) @ (pcl_frame.reshape(-1,3) - t).T/1000).T)
    pcl.colors = o3d.utility.Vector3dVector(color_frame.reshape(-1, 3)/255)
    return R, t, pcl


def run_kinect_calib_offline(kwargs: dict):
    """
    Compute R, t and colored pcl for a all cameras using mkv
    """
    output_path = os.path.join(kwargs["output"], "kinect")
    clean_dir(output_path)
    loader = KinectResultLoader(kwargs["input"], params=[
        dict(tag="kinect/master/color", ext=".png"),
        dict(tag="kinect/master/pcls", ext=".npy"),
        dict(tag="kinect/sub1/color", ext=".png"),
        dict(tag="kinect/sub1/pcls", ext=".npy"),
        dict(tag="kinect/sub2/color", ext=".png"),
        dict(tag="kinect/sub2/pcls", ext=".npy"),
    ])

    frame = loader.select_by_id(kwargs.get("frame_id", 80))

    aruco_size = float(kwargs.get("aruco_size", 0.8))

    devices = dict(master=MAS, sub1=SUB1, sub2=SUB2)

    results = dict()
    for device, idx in devices.items():
        color_frame = cv2.cvtColor(cv2.imread(frame["kinect/{}/color".format(device)]["filepath"]), cv2.COLOR_BGR2RGB)
        pcl_frame = np.load(frame["kinect/{}/pcls".format(device)]["filepath"])
        R, t, pcl = compute_single_transform(color_frame, pcl_frame, INTRINSIC[idx], aruco_size)
        results[device] = dict(R=R, t=t, pcl=pcl)
        o3d.io.write_point_cloud(os.path.join(output_path, "{}.ply".format(device)), pcl)
        np.savez(os.path.join(output_path, "{}".format(device)), R=R, t=t)

    o3d_plot([res["pcl"] for res in results.values()])
    return results


def run_kinect_calib_online(kwargs):
    """
    Compute R, t and colored pcl for a all cameras directly
    """
    from multiprocessing.dummy import Pool
    from pyk4a import PyK4A
    from dataloader.kinect_loader import _get_config, _get_device_ids
    
    output_path = os.path.join(kwargs["output"], "kinect")
    clean_dir(output_path)

    aruco_size = float(kwargs.get("aruco_size", 0.8))
    
    def process(device_type, device_id):
        device = PyK4A(config=_get_config(device_type), device_id=device_id)
        device.start()
        
        # skip first 60 frames
        for i in range(60):
            device.get_capture()
        
        capture = device.get_capture()
        # device.close()
        return cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB), capture.transformed_depth_point_cloud

    devices = dict(master=MAS, sub1=SUB1, sub2=SUB2)
    devices_id = _get_device_ids()
    devices_enabled = kwargs.get("devices", "master_sub1_sub2").split('_')
    devices_type = dict(master="mas", sub1="sub", sub2="sub")

    pool = Pool()
    results = dict()
    for device, idx in devices.items():
        # skip not enabled devices
        if device not in devices_enabled:
            continue
        results[device] = dict()
        results[device] = pool.apply_async(process, (devices_type[device], devices_id[idx])).get()
    pool.close()
    pool.join()

    for device, result in results.items():
        color_frame, pcl_frame = result
        R, t, pcl = compute_single_transform(color_frame, pcl_frame, INTRINSIC[idx], aruco_size)
        results[device] = dict(R=R, t=t, pcl=pcl)
        o3d.io.write_point_cloud(os.path.join(output_path, "{}.ply".format(device)), pcl)
        np.savez(os.path.join(output_path, "{}".format(device)), R=R, t=t)

    o3d_plot([res["pcl"] for res in results.values()])    
    return results

def run_null():
    """
    Default run function
    """
    raise RuntimeError("[calibration] Task must be specified.")


def run():
    task_dict = dict(
        null=run_null,
        kinect_offline=run_kinect_calib_offline,
        kinect_online=run_kinect_calib_online,
        kinect=run_kinect_calib_online,
        # optitrack=run_optitrack_calib,
    )
    parser = argparse.ArgumentParser(usage='"run_calibration.py -h" to show help.')
    parser.add_argument('-t', '--task', dest='task', type=str,
                        choices=list(task_dict.keys()), default='null', help='Run Target, default "null". {}'.format(task_dict))
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default='./ignoredata/calib_files/input', help='Input File Path, default "./ignoredata/calib_files/input"')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='', help='Output File Path, default $input + "calib"')
    parser.add_argument('-a', '--addition', dest='addition', type=str,
                        default='', help='Addition args split by "#", default ""')
    
    args = parser.parse_args()
    args_dict = dict([arg.split('=') for arg in args.addition.split('#')])
    args_dict.update(dict(args._get_kwargs()))
    if args_dict["output"] == "":
        args_dict["output"] = os.path.join(args_dict["input"], "calib")
    task_dict[args.task](args_dict)


if __name__ == "__main__":
    run()
