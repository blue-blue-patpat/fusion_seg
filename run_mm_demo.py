import os
import argparse
import time
import curses
import json
import numpy as np
from queue import Queue

from nn.p4t.recon_demo import ReconstructionDemo
from visualization.mesh_plot import MoshEvaluateStreamPlot
from nn.p4t.tools import copy2cpu as c2c
from visualization.utils import filter_pcl

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from dataloader.utils import MultiSubClient, ymdhms_time, print_log
from dataloader.arbe_loader import ArbeSubscriber, arbe_demo_callback
from nn.votenet.dete_demo import DetectionDemo
from sensor_msgs import point_cloud2

parser = argparse.ArgumentParser(usage='"run_dataloader.py -h" to show help.')
parser.add_argument('-e', '--env', dest='env', type=str, choices=[
                    'dev', 'prod'], default='dev', help='Environment, default "dev". dev: save in ./default; prod: save by timestamp')
parser.add_argument('-p', '--savepath', dest='save_path', type=str,
                    default='./__test__/', help='File Save Path, default "./__test__/"')
parser.add_argument('-nlog', '--disableLog', dest='disable_log',
                    action="store_true", help='Disable log, default False')
parser.add_argument('-npannel', '--disablePannel', dest='disable_pannel',
                    action="store_true", help='Disable pannel, default False')
parser.add_argument('-nvis', '--disableVisualization', dest='disable_visualization',
                    action="store_true", help='Disable visualization, default False')
parser.add_argument('-s', '--save', dest='save_data', default=False,
                    action="store_true", help='Save point cloud and mesh')
ARGS = parser.parse_args()

def run_arbe_loader(client=None, **kwargs):
    if client is None:
        client = MultiSubClient()

    client.add_sub(kwargs.get("name", "/arbe/rviz/pointcloud"), point_cloud2.PointCloud2, arbe_demo_callback,
                   sub_type=ArbeSubscriber,
                   save_path=kwargs.get("save_path", "./__test__/default/arbe/"),
                   log_obj=kwargs.get("log_obj", None),
                   disable_visualization=kwargs.get("disable_visualization", False),
                   save_data=kwargs.get("save_data", False),
                   queue=kwargs.get("queue", None),
                   )

def quit(client, **args):
    if args["env"] == "prod":
        import shutil
        args["log_obj"].close()
        shutil.rmtree(args["path"], ignore_errors=True)
    exit(0)

def result_generator(pcl, mesh):
    if mesh is not None:
        result = dict(
            radar_pcl = dict(pcl = pcl, color = [0,0.8,0],),
            pred_smpl = dict(
                mesh = [c2c(mesh['verts']), c2c(mesh['faces'])],
                color = np.asarray([235, 189, 191]) / 255,
            )
        )
    else:
        result = dict(radar_pcl = dict(pcl = pcl, color = [0,0.8,0],))
                
    yield result

def run():
    # log obj, None if log is disabled
    log_obj = None

    # default cmd, equals to last cmd
    default_cmd = None

    client = MultiSubClient()
    device = torch.device('cuda')
    dete_model = DetectionDemo(device)
    recon_model = ReconstructionDemo(device)
    plot = MoshEvaluateStreamPlot()
    # dataloader loop, ctrl+c to interrupt
    while True:
        # try closing log obj 
        try:
            log_obj.close()
        except:
            pass

        # decide save path
        if ARGS.env == "dev":
            root_path = os.path.join(ARGS.save_path, "default")
        else:
            root_path = os.path.join(ARGS.save_path, ymdhms_time())

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        # init log file obj
        if not ARGS.disable_log:
            log_path = os.path.join(root_path, "log.txt")
            log_obj = open(log_path, "a")
            print_log("[DataLoader] Logging enabled, check {} for more detailed information.".format(
                log_path), log_obj, always_console=True)

        print_log("[DataLoader] Running in {} env, save to path {}.".format(
            ARGS.env, root_path), log_obj, always_console=True)
        
        queue = Queue()
        # command list
        cmd_dict = dict(
            arbe=dict(
                key="a",
                name='/arbe/rviz/pointcloud',
                func=run_arbe_loader,
                args=dict(
                    save_path=os.path.join(root_path, "arbe"),
                    name='/arbe/rviz/pointcloud',
                    log_obj=log_obj,
                    disable_visualization=ARGS.disable_visualization,
                    save_data=ARGS.save_data,
                    queue=queue
                )
            ),
            default=dict(
                key="Enter",
                name="/Default",
            ),
            quit=dict(
                key="q",
                name="/QuitMainProcess",
                func=quit,
                args=dict(
                    path=root_path,
                    env=ARGS.env,
                    log_obj=log_obj
                )
            ),
        )

        # re-init client
        client.content_init()

        # show commands
        for k, v in cmd_dict.items():
            print("{} for {}".format(v["key"], k))

        
        # show default command
        if default_cmd:
            print("default: {}".format(
                [k for k, v in cmd_dict.items() if v["key"] in default_cmd]))

        # wait for command input
        cmd = input(
            "{} : [DataLoader] Eneter command to continue...\n".format(ymdhms_time()))

        # show chosen commands
        print_log("[DataLoader] Running with command {}.".format(
            cmd), log_obj, always_console=True)
        print_log("[DataLoader] Tasks: {}.".format(
            [k for k, v in cmd_dict.items() if v["key"] in cmd]), log_obj, always_console=True)

        # if command is just Enter, use default; else init default
        if cmd == "" and default_cmd:
            cmd = default_cmd
        else:
            default_cmd = cmd
                    
        # run command functions
        for k, v in cmd_dict.items():
            if v["key"] in cmd:
                v["func"](client, **v["args"])

        # start client's tasks
        client.start_all_subs()

        # show pannel if not disabled
        if not ARGS.disable_pannel:
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()

        try:
            # ks.start()
            # wait for tasks, leave loop when tasks are all stoped
            while len(client.get_running_tasks()) > 0:
                # update pannel info
                if not ARGS.disable_pannel:
                    client.print_info(stdscr)

                # refresh log file
                if log_obj:
                    log_obj.flush()
                if not queue.empty():
                    arbe_data = queue.get()
                    if arbe_data.shape[0]:
                        # print('Get valid points to run detection module.')
                        arbe_pcl = arbe_data[:, [1,2,3,4,8,9]]
                        dete_results = dete_model.run_detection(arbe_pcl, root_path, ARGS.save_data)
                        if dete_results is not None:
                            person_pcl = filter_pcl(dete_results, arbe_pcl, 0)
                            bbox_center = ((dete_results.max(axis=0) + dete_results.min(axis=0))/2)[:3]
                            pred_mesh = recon_model.run_reconstruction(person_pcl, bbox_center, root_path)
                        else:
                            person_pcl = np.asarray([0,0,0], dtype=np.float32)[None]
                            pred_mesh = None
                            # refresh the clip
                            recon_model.clip = []
                        gen = result_generator(person_pcl[:,:3], pred_mesh)
                        plot.show(gen, fps=30)
                else:
                    # sleep to avoid pannel flashing
                    time.sleep(0.1)

        except Exception as e:
            print_log(e, log_obj=log_obj, always_console=True)

        finally:
            # close pannel
            if not ARGS.disable_pannel:
                curses.echo()
                curses.nocbreak()
                curses.endwin()

        print_log("[DataLoader] Interrupted.", log_obj)

if __name__ == '__main__':
    run()
