import json
from math import inf
from multiprocessing import process
from multiprocessing.dummy import Pool
import os
import numpy as np
from dataloader.utils import clean_dir, ymdhms_time
from dataloader.result_loader import KinectResultLoader

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
