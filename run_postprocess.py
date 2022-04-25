from multiprocessing.dummy import Pool
from kinect.kinect_mkv import extract_mkv
from kinect.kinect_skeleton import extract_skeleton
from dataloader.result_loader import KinectMKVtLoader, OptitrackCSVLoader
import os
from optitrack.optitrack_loader import parse_opti_csv


def postprocess(root_path, devices, output_path=None):
    if not devices:
        devices = ["master","sub1","sub2"]
    if output_path is None:
        output_path = root_path
    params = [dict(tag="kinect/{}".format(device), ext=".mkv") for device in devices]
    pool = Pool()

    # if needs to parse the opti_csv
    try:
        csv_file = OptitrackCSVLoader(root_path)
        if len(csv_file):
            parse_opti_csv(csv_file.file_dict["optitrack"].loc[0,"filepath"])
    except:
        print('No optitrack csv!')

    def process(device):
        mkv_path = os.path.join(root_path, "kinect", device)
        os.system("ignoredata/kinect_files/offline_processor {}/out.mkv {}/out.json".format(mkv_path, mkv_path))
        try:
            extract_skeleton(root_path, device)
        except:
            print('No kinect skeleton!')

    for device in devices:
        pool.apply_async(process, (device,))
        # process(device)

    mkv_loader = KinectMKVtLoader(root_path, params)
    mkv_list = [m.loc[0,"filepath"] for m in mkv_loader.file_dict.values()]
    pool.map_async(extract_mkv, mkv_list)
    # for mkv in mkv_loader.mkvs.values():
    #     extract_mkv(mkv)

    pool.close()
    pool.join()


def visualize():
    pass


def run():
    import argparse
    task_dict = dict(
        null=exit,
        postprocess=postprocess,
    )
    parser = argparse.ArgumentParser(usage='"run_postptocess.py -h" to show help.')
    parser.add_argument('-p', '--path', dest='root_path', type=str, help='File Root Path, default "./__test__/default"')
    parser.add_argument('-d', '--device', dest='device', type=str,
                        default='master,sub1,sub2', help='Process Devices, default "master,sub1,sub2"')
    parser.add_argument('-t', '--task', dest='task', type=str,
                        choices=list(task_dict.keys()), default='postprocess', help='Run Target, default "null". {}'.format(task_dict))

    args = parser.parse_args()

    task_dict[args.task](args.root_path, args.device.split(','))

if __name__ == '__main__':
    run()
