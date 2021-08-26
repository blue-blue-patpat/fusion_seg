from multiprocessing.dummy import Pool
from time import sleep
from kinect.kinect_mkv import extract_mkv
from kinect.kinect_skeleton import extract_skeleton
from dataloader.result_loader import KinectMKVtLoader
import os


def postprocess(root_path, *devices):
    if not devices:
        devices = ["master","sub1","sub2"]
    params = []
    pool = Pool()

    def process(device):
        json_path = os.path.join(root_path, "kinect", device)
        os.system("ignoredata/kinect_files/offline_processor {}/out.mkv {}/out.json".format(json_path, json_path))
        extract_skeleton(root_path, device)

    for device in devices:
        params.append(dict(tag="kinect/{}".format(device), ext=".mkv"))
        pool.apply_async(process, (device,))
    mkvs = KinectMKVtLoader(root_path, params)
    mkv_list = [m.iloc[0,1] for m in mkvs.file_dict.values()]
    pool.map_async(extract_mkv, mkv_list)
    pool.close()
    pool.join()


def visualize():
    pass


def run():
    import argparse
    parser = argparse.ArgumentParser(usage='"run_postptocess.py -h" to show help.')
    parser.add_argument('-p', '--path', dest='root_path', type=str, help='File Root Path, default "./__test__/default"')
    parser.add_argument('-d', '--device', dest='device', type=str,
                        default='master,sub1,sub2', help='Process Devices, default "master,sub1,sub2"')
    args = parser.parse_args()

    postprocess(args.root_path, *args.device.split(','))


if __name__ == '__main__':
    run()
