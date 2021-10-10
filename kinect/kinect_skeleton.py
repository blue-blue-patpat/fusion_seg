import json
from math import inf
from multiprocessing import process
from multiprocessing.dummy import Pool
import os
import numpy as np
from dataloader.utils import clean_dir, ymdhms_time


def extract_skeleton(root_path, *devices):
    """
    Extract skeleton from kinect
    Parameters:
        parent_path: the parent path of kinect, such as "./2021-08-09_20-28-20"
        devices: the devices of kinect
    """
    if not devices:
        devices = ("master","sub1","sub2")
    for device in devices:
        file_path = os.path.join(root_path, "kinect", device)
        clean_dir(os.path.join(file_path, "skeleton"))
        with open(os.path.join(file_path, "out.json"),'r',encoding="utf8") as f:
            json_data = json.load(f)
        with open(os.path.join(file_path, "info.txt"), 'r') as f:
            params = dict([param.split('=') for param in f.readline().split('_')])
        info = [0, 0]
        pool = Pool()

        def process(save_path, bodies, info):
            np.save(save_path, bodies)
            info[0] += 1

        i = 0
        print()
        for frame in json_data["frames"]:
            bodies = []
            for body in frame["bodies"]:
                bodies.append(np.hstack((np.asarray(body["joint_positions"]), np.asarray(body["joint_orientations"]))))
            if bodies:
                filename = "id={}_skid={}_st={}_dt={}".format(frame["frame_id"], i, float(params["starttm"])+frame["timestamp_usec"]/1000000, float(params["tasktm"])+frame["timestamp_usec"]/1000000)
                save_path = os.path.join(file_path, "skeleton", filename)
                
                pool.apply_async(process, (save_path, bodies, info))
                print("{} : [Kinect skel] Extracting {}/{} frame.".format(ymdhms_time(), info[0], info[1]), end="\r")
                info[1] += 1
                i += 1
        pool.close()
        pool.join()
        print()


def extract_skeleton_v2(root_path, *devices):
    """
    Extract skeleton from kinect
    Parameters:
        parent_path: the parent path of kinect, such as "./2021-08-09_20-28-20"
        devices: the devices of kinect
    """
    import ctypes
    from kinect.k4a.kinectBodyTracker import kinectBodyTracker
    from kinect.k4a.exampleBodyTracking import modulePath, bodyTrackingModulePath
    from kinect.k4a import _k4a, _k4abt, _k4atypes, _k4arecord, _k4arecordTypes
    from pyk4a.calibration import Calibration
    from pyk4a.playback import PyK4APlayback
    from pyk4a.config import DepthMode, ColorResolution
    
    import cv2
    
    if not devices:
        devices = ("master","sub1","sub2")
    for device in devices:
        _k4a.k4a.setup_library(modulePath)

        k4a_record = _k4arecord.k4arecord(modulePath)
        capture_handle = _k4a.k4a_capture_t()
        calibration_handle = _k4a.k4a_calibration_t()

        playback_handle = _k4arecordTypes.k4a_playback_t()
        k4a_record.k4a_playback_open(ctypes.create_string_buffer(os.path.join(root_path, "kinect", device, "out.mkv").encode('utf-8')), playback_handle)
        
        # k4a.k4a_calibration_get_from_raw(ctypes.create_string_buffer(raw_calibration.encode("utf-8")), ctypes.c_size_t(len(raw_calibration)), _k4atypes.K4A_DEPTH_MODE_NFOV_UNBINNED, _k4atypes.K4A_COLOR_RESOLUTION_1536P, calibration_t)

        k4a_record.k4a_playback_get_calibration(playback_handle, calibration_handle)
        body_tracker = kinectBodyTracker(bodyTrackingModulePath, calibration_handle, _k4abt.K4ABT_DEFAULT_MODEL)
        while True:
            k4a_record.k4a_playback_get_next_capture(playback_handle, capture_handle)

            body_tracker.enqueue_capture(capture_handle)
            body_tracker.detectBodies()

            for body in body_tracker.bodiesNow:
                body_tracker.printBodyPosition(body)
                print(body.skeleton)
