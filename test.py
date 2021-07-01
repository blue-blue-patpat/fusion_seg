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


if __name__ == '__main__':
    arbe_loader_offline_test()
