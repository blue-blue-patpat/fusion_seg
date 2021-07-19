
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
    from dataloader.nokov_loader import nokov_loader_before_start, nokov_loader_callback, nokov_loader_after_stop

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


def arbe_loader_post_processing(sub: dict, **kwargs):
    import os
    from dataloader.utils import clean_dir

    save_path = sub["args"].get('path', './__test__/arbe_output/')
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


def kinect_loader_rgb(client=None, name="/rgb/image_raw"):
    from dataloader.kinect_loader import kinect_loader_before_start, kinect_loader_callback
    from sensor_msgs.msg import Image

    if client is None:
        client = MultiSubClient()
    
    client.add_sub(name, Image, kinect_loader_callback,
                    before_start=kinect_loader_before_start,
                    save_path="./__test__/kinect_output/rgb")


def kinect_loader_depth(client=None, name="/depth/image_raw"):
    from dataloader.kinect_loader import kinect_loader_before_start, kinect_loader_callback
    from sensor_msgs.msg import Image

    if client is None:
        client = MultiSubClient()
    
    client.add_sub(name, Image, kinect_loader_callback,
                    before_start=kinect_loader_before_start,
                    save_path="./__test__/kinect_output/depth")


def kinect_loader_multi(client=None, name="KinectSDK"):
    from dataloader.kinect_loader import KinectSubscriber
    from kinect.kinect_loader import _get_device_ids, _get_config
    from multiprocessing import Value

    if client is None:
        client = MultiSubClient()

    id_dict = _get_device_ids()
    print("[{}] {} devices found.".format(name, len(id_dict)))

    # kinect_master_release_flag = 

    client.add_sub("KinectSub2", sub_type=KinectSubscriber, config=_get_config("sub"),
                                device_id=id_dict["000176712112"],
                                save_path="./__test__/kinect_output/sub2")
    client.start_sub("KinectSub2")

    client.add_sub("KinectSub1", sub_type=KinectSubscriber, config=_get_config("sub"), 
                                device_id=id_dict["000053612112"], 
                                save_path="./__test__/kinect_output/sub1")
    client.start_sub("KinectSub1")

    client.add_sub("KinectMaster", sub_type=KinectSubscriber, config=_get_config("mas"),
                                device_id=id_dict["000326312112"],
                                save_path="./__test__/kinect_output/master")
    client.start_sub("KinectMaster")


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
        kinect_multi=dict(
            key="k0",
            name = "/KinectSDK",
            func=kinect_loader_multi,
            post_func=None
        ),
        kinect_single_rgb=dict(
            key="k1",
            name = "/rgb/image_raw",
            func=kinect_loader_rgb,
            post_func=None
        ),
        kinect_singel_depth=dict(
            key="k2",
            name = "/depth/image_raw",
            func=kinect_loader_depth,
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
        print("Eneter command to continue...")
        client = MultiSubClient()

        for k, v in cmd_dict.items():
            print("{} for {}".format(v["key"], k))
        cmd = input()
        for k, v in cmd_dict.items():
            if v["key"] in cmd:
                v["func"](client)

        # client.start_all_subs()

        try:
            while True:
                pass
        except TaskFinishException as e:
            print(e)

        for k, v in cmd_dict.items():
            if v["key"] in cmd and v["post_func"] is not None:
                v["post_func"](client.subscribers[v["name"]])
