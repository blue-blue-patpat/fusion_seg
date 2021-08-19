def data_aling_test():
    from dataloader import align
    df = align('./dataloader/__test__/gt', './dataloader/__test__/camera',
               './dataloader/__test__/radar', './dataloader/__test__/output/test.csv', abspath=False)
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


def minimal_test():
    import os
    import torch
    from pytorch3d.structures import Meshes, Pointclouds
    from minimal.solver import Solver
    from minimal import armatures
    from minimal.models import KinematicModel, KinematicPCAWrapper
    import numpy as np
    import minimal.config as config
    from dataloader.result_loader import KinectResultLoader
    from minimal.bridge import JointsBridge

    from dataloader.utils import ymdhms_time

    np.random.seed(20160923)
    # pose_glb = np.zeros([1, 3]) # global rotation

    n_pose = 23 * 3 # degrees of freedom, (n_joints - 1) * 3
    # smpl 1.0.0: 10
    # smpl 1.1.0: 300
    n_shape = 10

    k_loader = KinectResultLoader('/media/nesc525/perple/2021-08-18_10-49-44-T')
    files = k_loader.select_item(298, "id")
    kinect_skeleton = np.load(files["kinect/master/skeleton"]["filepath"])
    kinect_pcls = np.vstack(np.load(files['kinect/master/pcls']["filepath"]))

    bridge = JointsBridge()
    keypoints_gt, pcl_gt, origin_scale = bridge.smpl_from_kinect(kinect_skeleton[0], kinect_pcls)

    mesh = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature, scale=1)

    wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
    solver = Solver(wrapper, max_iter=int(10e7))

    # pointcloud_gt = None
    params_est = solver.solve_full(keypoints_gt, pcl_gt, origin_scale, verbose=5)

    shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

    print('----------------------------------------------------------------------')
    print('estimated parameters')
    print('pose pca coefficients:', pose_pca_est)
    print('pose global rotation:', pose_glb_est)
    print('shape: pca coefficients:', shape_est)

    mesh.set_params(pose_pca=pose_pca_est)
    mesh.save_obj(os.path.join(config.SAVE_PATH, './esttm={}.obj'.format(ymdhms_time())))


def result_loader_test(file_path):
    from dataloader.result_loader import KinectResultLoader
    k_loader = KinectResultLoader(file_path)
    gen = k_loader.generator()
    print(next(gen))
    # print(k_loader.select_item(1627738564.7240, 'st', exact=False))
    # print(k_loader[3])

def result_manager_test(filepath):
    import numpy as np
    from visualization.pcd_visual import SkelArbeManager
    rm = SkelArbeManager(filepath)



if __name__ == "__main__":
    minimal_test()
    # result_manager_test('/home/nesc525/chen/3DSVC/__test__/2021-08-05 17:21:35')
    # result_loader_test('./__test__/mkv/')

    # from kinect.kinect_mkv import extract_mkv
    # extract_mkv("/media/nesc525/perple/2021-08-09_20-28-20/kinect/sub2/tasktm=1628512119.258128.mkv", False) 

    # from visualization import pcd_visualization
    # from multiprocessing import Process
    # parent_path = "/media/nesc525/perple/2021-08-09_19-47-45"
    # visual_front = Process(target=pcd_visualization, args=(parent_path, "master"))
    # visual_left = Process(target=pcd_visualization, args=(parent_path, "master"))
    # visual_right = Process(target=pcd_visualization, args=(parent_path, "master"))
    # visual_front.start()
    # visual_left.start()
    # visual_right.start()
    # visual_front.join()
    # visual_left.join()
    # visual_right.join()

    # pcd_visualization("/media/nesc525/perple/2021-08-09_20-28-20", "master")
    # pcd_visualization("/home/nesc525/chen/3DSVC/__test__/2021-08-06 14:09:37")

    # from minimal.bridge import JointsBridge
    # import numpy as np
    # np.save("ignoredata/minimal_files/output/id=200_tm=60127922_st=1627651318.2057362.npy", bridge.smpl_jnts)

    # from dataloader.result_loader import KinectResultLoader
    # from minimal.bridge import JointsBridge
    # import numpy as np
    # from visualization.pcd_visual import vis_smpl_skeleton

    # k_loader = KinectResultLoader('./ignoredata/minimal_files/input/')
    # files = k_loader.select_by_id(200)
    # kinect_skeleton = np.load(files["kinect/master/skeleton"]["filepath"])

    # bridge = JointsBridge()
    # bridge.load_kinect_joints(kinect_skeleton[0])
    # bridge.kinect_joints_transfer_coordinates()
    # keypoints_gt = bridge.update_smpl_joints_from_kinect_joints()

    # vis_smpl_skeleton(keypoints_gt)
    # vis_smpl_skeleton(np.load("/home/nesc525/chen/3DSVC/__test__/mkv/kinect/master/skeleton/id=165_st=1628392679.8134906_dt=1628392680.1205788.npy"))

    # from kinect.kinect_skeleton import extract_skeleton
    # extract_skeleton("/media/nesc525/perple/2021-08-09_20-28-20")
