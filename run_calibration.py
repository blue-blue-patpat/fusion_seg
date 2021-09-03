import os
import argparse
import cv2
import numpy as np
import open3d as o3d
from kinect.calib import calibrate_kinect
from kinect.config import MAS, SUB1, SUB2, INTRINSIC
from dataloader.utils import clean_dir
from dataloader.result_loader import KinectResultLoader
from visualization.utils import o3d_plot, o3d_pcl, o3d_coord


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
    _R = np.linalg.inv(R)
    _t = -_R @ t.T/1000
    pcl = o3d_pcl(pcl_frame.reshape(-1,3)/1000 @ _R.T + _t)
    pcl.colors = o3d.utility.Vector3dVector(color_frame.reshape(-1, 3)/255)
    return _R, _t, pcl


def run_kinect_calib_cpp(root_path, devices:list=["master","sub1","sub2"]):
    import json
    trans_mat = {}
    num_dict = {"master":0, "sub1":1, "sub2":2}
    for d in devices:
        with open(root_path+'/calib/kinect/matrix{}.json'.format(num_dict[d]),'r',encoding='utf8') as f:
            json_data = json.load(f)
        trans_mat[d] = np.asarray(list(json_data['value0']['matrix'].values()), np.float64).reshape(4,4)
    return trans_mat


def run_kinect_viewer(kwargs: dict):
    from kinect.aruco import detect_aruco
    detect_aruco()


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

    devices = dict(master=MAS, sub1=SUB1, sub2=SUB2)
    devices_id = _get_device_ids()
    devices_enabled = kwargs.get("devices", "master_sub1_sub2").split('_')
    devices_type = dict(master="mas", sub1="sub", sub2="sub")

    started_devices = dict()
    for device, idx in devices.items():
        # skip not enabled devices
        if device not in devices_enabled:
            continue
        started_devices[device] = PyK4A(config=_get_config(devices_type[device]), device_id=devices_id[idx])
        started_devices[device].start()

    results = dict()

    for i in range(60):
        for device, idx in devices.items():
            # skip not enabled devices
            if device not in devices_enabled:
                continue
            # skip first 60 frames
            started_devices[device].get_capture()

    print("[Kinect Calibrate Online] Start calibration...")
    has_result = False

    while not has_result:
        try:
            for device, idx in devices.items():
                if device not in devices_enabled:
                    continue
                capture = started_devices[device].get_capture()

                color_frame, pcl_frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB), capture.transformed_depth_point_cloud

                R, t, pcl = compute_single_transform(color_frame, pcl_frame, INTRINSIC[idx], aruco_size)
                results[device] = dict(R=R, t=t, pcl=pcl)
                o3d.io.write_point_cloud(os.path.join(output_path, "{}.ply".format(device)), pcl)
                np.savez(os.path.join(output_path, "{}".format(device)), R=R, t=t)
            o3d_plot([res["pcl"] for res in results.values()])
            confirm_info = input("Do you want to save current result?\n[y/n]\n")
            if 'y' in confirm_info:
                has_result = True
        except Exception as e:
            print(e)
            print("[Kinect Calibrate Online] Retry calibration...")

    return results


def run_optitrack_calib(kwargs=None, root_path="/__test__/default"):
    from dataloader.result_loader import OptitrackCalibResultLoader
    from optitrack.calib import read_csv, trans_matrix

    output_path = os.path.join(kwargs.get("output",root_path+"/calib/"), "optitrack")
    clean_dir(output_path)
    loader = OptitrackCalibResultLoader(kwargs.get("input", root_path))

    coords = read_csv(loader.file_dict["calib/input"].iloc[0]["filepath"])
    R, t = trans_matrix(coords)

    np.savez(os.path.join(output_path, "transform"), R=R, t=t)

    o3d_plot([o3d_pcl(coords @ R.T + t)], size=0.1)
    return R, t


def run_null():
    """
    Default run function
    """
    raise RuntimeError("[calibration] Task must be specified.")


def run():
    task_dict = dict(
        null=run_null,
        # kinect viewer
        kinect_viewer=run_kinect_viewer,
        # kinect offline
        kinect_offline=run_kinect_calib_offline,
        # kinect online
        kinect_online=run_kinect_calib_online,
        kinect=run_kinect_calib_online,
        k=run_kinect_calib_online,
        # optitrack
        optitrack=run_optitrack_calib,
        opti=run_optitrack_calib,
        o=run_optitrack_calib,
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
    args_dict = dict([arg.split('=') for arg in args.addition.split('#') if '=' in arg])
    args_dict.update(dict(args._get_kwargs()))
    if args_dict["output"] == "":
        args_dict["output"] = os.path.join(args_dict["input"], "calib")
    task_dict[args.task](args_dict)


if __name__ == "__main__":
    run()
