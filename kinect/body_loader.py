import json
import os
import numpy as np
from dataloader.utils import clean_dir
from dataloader.result_loader import KinectResultLoader

def extract_body(parent_path, device="master"):
    file_path = os.path.join(parent_path, "kinect", device)
    clean_dir(os.path.join(file_path, "skeleton"))
    with open(os.path.join(file_path, "out.json"),'r',encoding="utf8") as f:
        json_data = json.load(f)
    with open(os.path.join(file_path, "info.txt"), 'r') as f:
        params = dict([param.split('=') for param in f.readline().split('_')])
    kr = KinectResultLoader(parent_path, device=device)
    for frame in json_data["frames"]:
        bodies = []
        for body in frame["bodies"]:
            bodies.append(body["joint_positions"])
        # kinect_item = kr.select_by_id(frame["frame_id"])["filename"]
        # dir, _ = os.path.split(kinect_item)
        # filename, extension = os.path.splitext(_)
        filename = "id={}".format(frame["frame_id"])
        if bodies:
            save_path = os.path.join(file_path, "skeleton", "{}_tm={}".format(filename, frame["timestamp_usec"]))
            np.save(save_path, bodies)

if __name__ == "__main__":
    extract_body("/home/nesc525/chen/3DSVC/__test__/mkv")