from multiprocessing.dummy import Pool
import optitrack
from time import sleep
from kinect.kinect_mkv import extract_mkv
from kinect.kinect_skeleton import extract_skeleton
from dataloader.result_loader import KinectMKVtLoader, OptitrackCSVLoader
import os
from optitrack.optitrack_loader import parse_opti_csv
from run_calibration import run_optitrack_calib

def postprocess(root_path, *devices, calib=False):
    if not devices:
        devices = ["master","sub1","sub2"]
    params = []
    pool = Pool()

    def process(device):
        mkv_path = os.path.join(root_path, "kinect", device)
        os.system("ignoredata/kinect_files/offline_processor {}/out.mkv {}/out.json".format(mkv_path, mkv_path))
        extract_skeleton(root_path, device)

    for device in devices:
        params.append(dict(tag="kinect/{}".format(device), ext=".mkv"))
        pool.apply_async(process, (device,))
    mkvs = KinectMKVtLoader(root_path, params)
    mkv_list = [m.loc[0,"filepath"] for m in mkvs.file_dict.values()]
    pool.map_async(extract_mkv, mkv_list)
    # if needs to calibrate kinect and optitrack
    if calib:
        os.system("ignoredata/kinect_files/calib/calib_k4a {path}/kinect/master/out.mkv {path}/kinect/sub1/out.mkv {path}/kinect/sub2/out.mkv".format(path=root_path))
        os.system("mv *.ply ignoredata/kinect_files/calib/")
        os.system("mv {path}/kinect/master/matrix0.json {path}/calib/kinect".format(path=root_path))
        os.system("mv {path}/kinect/sub1/matrix1.json {path}/calib/kinect".format(path=root_path))
        os.system("mv {path}/kinect/sub2/matrix2.json {path}/calib/kinect".format(path=root_path))
        run_optitrack_calib(root_path=root_path)

    # if needs to parse the opti_csv
    try:
        csv_file = OptitrackCSVLoader(root_path)
        if len(csv_file):
            parse_opti_csv(csv_file.file_dict["optitrack"].loc[0,"filepath"])
    except:
        pass
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
    parser.add_argument('-c', '--calib', dest='calib', type=int, default=0, help='Calibrate the kinect if need, 1 or 0')
    args = parser.parse_args()

    postprocess(args.root_path, *args.device.split(','), calib=bool(args.calib))


if __name__ == '__main__':
    run()
