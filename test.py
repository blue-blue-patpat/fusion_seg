from dataloader.arbe_loader import arbe_loader_callback
import numpy as np


def data_aling_test():
    from dataloader import align
    df = align('./dataloader/__test__/gt', './dataloader/__test__/camera', './dataloader/__test__/radar', './dataloader/__test__/output/test.csv', abspath=False)
    print(df)


def coord_trans_test():
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


def nokov_loader_test():
    import rospy
    from dataloader import MultiSubClient, NOKOVLoader
    client = MultiSubClient()
    nokov_loader = NOKOVLoader()
    nokov_loader.init_dataframe()
    nokov_loader.init_client("/vrpn_client_node/tb3_0/pose", client=client)
    nokov_loader.start_sub()
    while not rospy.is_shutdown():
        if len(nokov_loader.get_dataframe()) > 20:
            nokov_loader.stop_sub()
            break
    df = nokov_loader.get_dataframe()
    print(df.head(10))
    df.to_csv('./dataloader/__test__/nokov_test.csv')


def arbe_loader_online_test():
    import rospy
    import pandas as pd
    from dataloader import MultiSubClient
    from sensor_msgs import point_cloud2
    name = '/arbe/rviz/pointcloud'
    client = MultiSubClient()
    
    client.add_sub(name, point_cloud2.PointCloud2, arbe_loader_callback)
    client.update_args(name, dict(dataframe=pd.DataFrame(), frame_id=0))
    client.start_sub(name)
    i = 0
    while not rospy.is_shutdown() and client.subscribers[name]['args']['frame_id']<5:
        pass
    client.stop_all_subs()
    print(client.subscribers[name]['args']['dataframe'])


if __name__ == '__main__':
    arbe_loader_online_test()
    # arbe_loader_offline_test()
