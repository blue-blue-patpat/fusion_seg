

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

    name = "/vrpn_client_node/tb3_0/pose"
    if client is None:
        client = MultiSubClient()

    client.add_sub(name, PoseStamped, nokov_loader_callback,
                    before_start=nokov_loader_before_start,
                    after_stop=nokov_loader_after_stop)
    while not rospy.is_shutdown():
        pass
    #     if len(nokov_loader.get_dataframe()) > 20:
    #         nokov_loader.stop_sub()
    #         break
    # df = nokov_loader.get_dataframe()
    # print(df.head(10))
    # df.to_csv('./dataloader/__test__/nokov_test.csv')


def arbe_loader_online_test(client=None):
    import rospy
    import os
    from dataloader.utils import MultiSubClient, clean_dir
    from dataloader.arbe_loader import arbe_loader_callback, arbe_loader_before_start, arbe_loader_after_stop
    from sensor_msgs import point_cloud2

    name = '/arbe/rviz/pointcloud'
    if client is None:
        client = MultiSubClient()
    
    client.add_sub(name, point_cloud2.PointCloud2, arbe_loader_callback,
                    before_start=arbe_loader_before_start, 
                    after_stop=arbe_loader_after_stop,
                    force_realtime=True)

    client.start_sub(name)

    while not rospy.is_shutdown():
        pass
    client.stop_all_subs()
    print(list(client.subscribers[name]['args']['dataframe'].values())[0])
    
    dir = './__test__/arbe_output/'
    clean_dir(dir)
    for ts, df in client.subscribers[name]['args']['dataframe'].items():
        df.to_csv(os.path.join(dir, '{}.csv'.format(ts)))
    # client.subscribers[name]['args']['dataframe'].to_csv('./dataloader/__test__/arbe_test.csv')


def realsense_loader_test(client=None):
    from dataloader.realsense_loader import RealSenseSubscriber
    from dataloader.utils import MultiSubClient

    name = "RealSense"

    if client is None:
        client = MultiSubClient()

    client.add_sub(name, sub_type=RealSenseSubscriber, save_path="./__test__/realsense_output")

    client.start_sub(name)
    try:
        while True:
            pass
    except KeyboardInterrupt:
        client.stop_sub(name)


if __name__ == '__main__':
    realsense_loader_test()
    # arbe_loader_online_test()
    # arbe_loader_offline_test()
