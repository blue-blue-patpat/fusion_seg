

from functools import cmp_to_key
from time import sleep
import dataloader
from dataloader.nokov_loader import nokov_loader_after_stop
from dataloader import utils


def data_aling_test():
    from dataloader import align
    df = align('./dataloader/__test__/gt', './dataloader/__test__/camera', './dataloader/__test__/radar', './dataloader/__test__/output/test.csv', abspath=False)
    print(df)


def coord_trans_test():
    import numpy as np
    from calib import coord_trans
    coords = coord_trans.read_static_ts('./calib/__test__/radar1.ts')
    # set or view board params here
    print(coord_trans.set_params())

    # compute transform matrix, print T, R param
    R, t = coord_trans.trans_matrix(coords)
    print(R, t)
    orig_coord = np.array([1000, 1500, 900])

    # transform a coordinate
    trans_coord = coord_trans.trans_coord(orig_coord, R, t)
    print(trans_coord)


def arbe_loader_offline_test():
    from dataloader import arbe_loader_offline
    df = arbe_loader_offline("2021-06-17-10-25-29.bag")
    print(df.head(10))
    # please delete test file after test
    df.to_csv('./dataloader/__test__/arbe_test.csv', index=False)


def nokov_loader_test(client=None):
    import rospy
    from dataloader.utils import MultiSubClient
    from geometry_msgs.msg import PoseStamped
    from dataloader.nokov_loader import nokov_loader_before_start, nokov_loader_callback

    name = "/vrpn_client_node/HelenHayes/pose"
    if client is None:
        client = MultiSubClient()

    client.add_sub(name, PoseStamped, nokov_loader_callback,
                    # sub_type=rospy.Subscriber,
                    before_start=nokov_loader_before_start,
                    after_stop=nokov_loader_after_stop)
    client.start_sub(name)
    while not rospy.is_shutdown():
        pass
    client.stop_all_subs()
    #     if len(nokov_loader.get_dataframe()) > 20:
    #         nokov_loader.stop_sub()
    #         break
    # df = nokov_loader.get_dataframe()
    # print(df.head(10))
    # df.to_csv('./dataloader/__test__/nokov_test.csv')


def arbe_loader(client=None, name = '/arbe/rviz/pointcloud'):
    from dataloader.utils import MultiSubClient
    from dataloader.arbe_loader import arbe_loader_callback, arbe_loader_before_start, arbe_loader_after_stop
    from sensor_msgs import point_cloud2

    if client is None:
        client = MultiSubClient()
    
    client.add_sub(name, point_cloud2.PointCloud2, arbe_loader_callback,
                    before_start=arbe_loader_before_start, 
                    after_stop=arbe_loader_after_stop,
                    force_realtime=True)

    client.start_sub(name)


def arbe_loader_post_processing(sub: dict, **kwargs):
    import os
    from dataloader.utils import clean_dir

    save_path = kwargs.get('path', './__test__/arbe_output/')
    clean_dir(save_path)
    for ts, df in sub['args']['dataframe'].items():
        df.to_csv(os.path.join(save_path, '{}.csv'.format(ts)))


def realsense_loader(client=None, name = "RealSense"):
    import os
    from dataloader.realsense_loader import RealSenseSubscriber
    from dataloader.utils import MultiSubClient

    if client is None:
        client = MultiSubClient()

    client.add_sub(name, sub_type=RealSenseSubscriber, save_path="./__test__/realsense_output")

    client.start_sub(name)


if __name__ == '__main__':
    from dataloader.utils import MultiSubClient, TaskFinishException
    print("DataLoader Test")
    cmd_dict = dict(
        arbe=dict(
            key="a",
            name = '/arbe/rviz/pointcloud',
            func=arbe_loader,
            post_func=arbe_loader_post_processing,
        ),
        realsense=dict(
            key="r",
            name = "/RealSense",
            func=realsense_loader,
            post_func=None
        ),
        quit=dict(
            key="q",
            name="/QuitMainProcess",
            func=exit,
            post_func=None
        ),
        # nokov=dict(
        #     key="n",
        #     func=nokov_loader,
        #     post_func=nokov_loader_post_processing
        # ),
    )
    while True:
        print("Press key to continue...")
        client = MultiSubClient()

        for k, v in cmd_dict.items():
            print("{} for {}".format(v["key"], k))
        cmd = input()
        for k, v in cmd_dict.items():
            if v["key"] in cmd:
                v["func"](client)

        try:
            while True:
                pass
        except TaskFinishException as e:
            print(e)

        for k, v in cmd_dict.items():
            if v["key"] in cmd and v["post_func"] is not None:
                v["post_func"](client.subscribers[v["name"]])
