#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   run_dataloader.py
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/7/23 18:46    wxy        1.0         use dataloader module to capture data
"""

def run_arbe_loader(client=None, **kwargs):
    from dataloader.utils import MultiSubClient
    from dataloader.arbe_loader import arbe_loader_callback, ArbeSubscriber
    from sensor_msgs import point_cloud2

    if client is None:
        client = MultiSubClient()

    client.add_sub(kwargs.get("name", "/arbe/rviz/pointcloud"), point_cloud2.PointCloud2, arbe_loader_callback,
                   sub_type=ArbeSubscriber,
                   save_path=kwargs.get("save_path", "./__test__/default/arbe/"),
                   log_obj=kwargs.get("log_obj", None),
                   disable_visualization=kwargs.get("disable_visualization", False))


def run_realsense_loader(client=None, **kwargs):
    from dataloader.utils import MultiSubClient
    from dataloader.realsense_loader import RealSenseSubscriber
    
    if client is None:
        client = MultiSubClient()

    client.add_sub(kwargs.get("name", "/RealSense"), sub_type=RealSenseSubscriber,
                   save_path=kwargs.get("save_path", "./__test__/default/realsense/"),
                   log_obj=kwargs.get("log_obj", None),
                   disable_visualization=kwargs.get("disable_visualization", False))


def run_kinect_loader_multi(client=None, **kwargs):
    import os
    from dataloader.utils import MultiSubClient, print_log
    from dataloader.kinect_loader import KinectSubscriber, KinectSkeletonSubscriber, _get_device_ids, _get_config
    
    MAS = "000326312112"
    SUB1 = "000053612112"
    SUB2 = "000176712112"

    if client is None:
        client = MultiSubClient()

    log_obj = kwargs.get("log_obj", None)

    id_dict = _get_device_ids()
    print_log("[{}] {} devices found.".format(kwargs.get("name", "KinectMulti"), len(id_dict)), log_obj)

    # first start two sub devices
    client.add_sub("KinectSub2", sub_type=KinectSubscriber, config=_get_config("sub"),
                   device_id=id_dict[SUB2],
                   save_path=os.path.join(kwargs.get(
                       "save_path", "./__test__/default/kinect/"), "sub2"),
                   log_obj=log_obj,
                   disable_visualization=kwargs.get("disable_visualization", False))
    client.start_sub("KinectSub2")

    client.add_sub("KinectSub1", sub_type=KinectSubscriber, config=_get_config("sub"),
                   device_id=id_dict[SUB1],
                   save_path=os.path.join(kwargs.get(
                       "save_path", "./__test__/default/kinect/"), "sub1"),
                   log_obj=log_obj,
                   disable_visualization=kwargs.get("disable_visualization", False))
    client.start_sub("KinectSub1")

    # then start the master divice
    client.add_sub("KinectMaster", sub_type=KinectSubscriber, config=_get_config("mas"),
                   device_id=id_dict[MAS],
                   save_path=os.path.join(kwargs.get(
                       "save_path", "./__test__/default/kinect/"), "master"),
                   log_obj=log_obj,
                   disable_visualization=kwargs.get("disable_visualization", False))
    client.start_sub("KinectMaster")
    
    # first start two sub devices
    # client.add_sub("KinectSub2", sub_type=KinectSkeletonSubscriber, config=_get_config("skeleton_sub"),
    #                device_id=id_dict.get(SUB2, 2),
    #                save_path=os.path.join(kwargs.get(
    #                    "save_path", "./__test__/default/kinect/"), "sub2"),
    #                log_obj=log_obj,
    #                disable_visualization=kwargs.get("disable_visualization", False))
    # client.start_sub("KinectSub2")

    # client.add_sub("KinectSub1", sub_type=KinectSkeletonSubscriber, config=_get_config("skeleton_sub"),
    #                device_id=id_dict.get(SUB1 ,1),
    #                save_path=os.path.join(kwargs.get(
    #                    "save_path", "./__test__/default/kinect/"), "sub1"),
    #                log_obj=log_obj,
    #                disable_visualization=kwargs.get("disable_visualization", False))
    # client.start_sub("KinectSub1")

    # client.add_sub("KinectMaster", sub_type=KinectSkeletonSubscriber, config=_get_config("skeleton_mas"),
    #                device_id=id_dict.get(MAS, 0),
    #                save_path=os.path.join(kwargs.get(
    #                    "save_path", "./__test__/default/kinect/"), "master"),
    #                log_obj=log_obj,
    #                disable_visualization=kwargs.get("disable_visualization", False))
    # client.start_sub("KinectMaster")

def run():
    import time
    import os
    import curses
    from dataloader.utils import MultiSubClient, ymdhms_time, print_log
    from dataloader.kinect_loader import KinectSkeletonSubscriber
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(usage='"start_dataloader.py -h" to show help.')
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
    args = parser.parse_args()
    # log obj, None if log is disabled
    log_obj = None

    # default cmd, equals to last cmd
    default_cmd = None

    client = MultiSubClient()

    # dataloader loop, ctrl+c to interrupt
    while True:
        # try closing log obj 
        try:
            log_obj.close()
        except:
            pass

        # decide save path
        if args.env == "dev":
            parent_path = os.path.join(args.save_path, "default")
        else:
            parent_path = os.path.join(args.save_path, ymdhms_time())

        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        # init log file obj
        if not args.disable_log:
            log_path = os.path.join(parent_path, "log.txt")
            log_obj = open(log_path, "a")
            print_log("[DataLoader] Logging enabled, check {} for more detailed information.".format(
                log_path), log_obj, always_console=True)

        print_log("[DataLoader] Running in {} env, save to path {}.".format(
            args.env, parent_path), log_obj, always_console=True)

        # command list
        cmd_dict = dict(
            arbe=dict(
                key="a",
                name='/arbe/rviz/pointcloud',
                func=run_arbe_loader,
                args=dict(
                    save_path=os.path.join(parent_path, "arbe"),
                    name='/arbe/rviz/pointcloud',
                    log_obj=log_obj,
                    disable_visualization=args.disable_visualization
                )
            ),
            realsense=dict(
                key="r",
                name="RealSense",
                func=run_realsense_loader,
                args=dict(
                    save_path=os.path.join(parent_path, "realsense"),
                    name='RealSense',
                    log_obj=log_obj,
                    disable_visualization=args.disable_visualization
                )
            ),
            kinect_multi=dict(
                key="k",
                name="KinectSDK",
                func=run_kinect_loader_multi,
                args=dict(
                    save_path=os.path.join(parent_path, "kinect"),
                    name='KinectSDK',
                    log_obj=log_obj,
                    disable_visualization=args.disable_visualization
                )
            ),
            default=dict(
                key="Enter",
                name="/Default",
            ),
            quit=dict(
                key="q",
                name="/QuitMainProcess",
                func=exit,
                args={}
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

        # ks = KinectSkeletonSubscriber()

        # show pannel if not disabled
        if not args.disable_pannel:
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()

        try:
            # ks.start()
            # wait for tasks, leave loop when tasks are all stoped
            while len(client.get_running_tasks()) > 0:
                # update pannel info
                if not args.disable_pannel:
                    client.print_info(stdscr)

                # refresh log file
                if log_obj:
                    log_obj.flush()
                # sleep to avoid pannel flashing
                time.sleep(0.1)

        except Exception as e:
            print_log(e, log_obj=log_obj, always_console=True)

        finally:
            # close pannel
            if not args.disable_pannel:
                curses.echo()
                curses.nocbreak()
                curses.endwin()

        print_log("[DataLoader] Interrupted.", log_obj)


if __name__ == '__main__':
    run()
