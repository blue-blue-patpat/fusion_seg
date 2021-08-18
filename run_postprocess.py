from multiprocessing.dummy import Pool
from time import sleep
from kinect.kinect_mkv import extract_mkv
from kinect.kinect_skeleton import extract_skeleton
from dataloader.result_loader import KinectMKVtLoader
import os

def postprocess(parent_path, *devices):
    if not devices:
        devices = ("master","sub1","sub2")
    params = []
    pool = Pool()

    def process(device):
        json_path = os.path.join(parent_path, "kinect", device)
        os.system("kinect/offline_processor {}/out.mkv {}/out.json".format(json_path, json_path))
        extract_skeleton(parent_path, device)

    for device in devices:
        params.append(dict(tag="kinect/{}".format(device), ext=".mkv"))
        pool.apply_async(process, (device,))
    mkvs = KinectMKVtLoader(parent_path, params)
    mkv_list = [m.iloc[0,1] for m in mkvs.file_dict.values()]
    pool.map_async(extract_mkv, mkv_list)
    pool.close()
    pool.join()

def visualize():
    pass

if __name__ == '__main__':
    parent_path = "/media/nesc525/perple/2021-08-09_19-47-45"
    # postprocess("/home/nesc525/chen/3DSVC/__test__/default", "sub1", "sub2")
    # extract_skeleton("/home/nesc525/chen/3DSVC/__test__/default", "sub1", "sub2")
    extract_skeleton(parent_path)
    # devices = ("master","sub1","sub2")
    # params = []
    # pool = Pool()
    # for device in devices:
    #     params.append(dict(tag="kinect/{}".format(device), ext=".mkv"))
    # mkvs = KinectMKVtLoader(parent_path, params)
    # mkv_list = [m.iloc[0,1] for m in mkvs.file_dict.values()]
    # pool.map_async(extract_mkv, mkv_list)
    # pool.close()
    # pool.join()