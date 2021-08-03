from pytorch3d.structures import pointclouds


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
    from pytorch3d.structures import Meshes, Pointclouds
    from minimal.solver import Solver
    from minimal import armatures
    from minimal.models import KinematicModel, KinematicPCAWrapper
    import numpy as np
    import minimal.config as config
    from dataloader.result_loader import KinectResultLoader
    from minimal.bridge import JointsBridge

    np.random.seed(20160923)
    # pose_glb = np.zeros([1, 3]) # global rotation


    ########################## smpl settings ##########################
    # note that in smpl and smpl-h no pca for pose is provided
    # therefore in the model we fake an identity matrix as the pca coefficients
    # to make the code compatible

    n_pose = 23 * 3 # degrees of freedom, (n_joints - 1) * 3
    # smpl 1.0.0: 10
    # smpl 1.1.0: 300
    # n_shape = 10
    # TODO: Read pose from skeleton
    k_loader = KinectResultLoader('./ignoredata/minimal_files/input/')
    files = k_loader.select_by_id(200)
    kinect_skeleton = np.load(files["master/skeleton"]["filepath"][0])
    joints = kinect_skeleton[0][:,:3]

    bridge = JointsBridge()
    bridge.load_kinect_joints(joints)
    keypoints_gt = bridge.update_smpl_joints()

    # pose_pca = np.random.uniform(-0.2, 0.2, size=n_pose)
    # shape = np.random.normal(size=n_shape)
    mesh = KinematicModel(config.SMPL_MODEL_1_0_PATH, armatures.SMPLArmature, scale=100)

    ########################## solving example ############################

    wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
    solver = Solver(verbose=True, max_iter=2000)

    # mesh_gt, keypoints_gt = \
    # mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
    # pointcloud_gt = Pointclouds(sample_points_from_meshes(mesh_gt, num_samples=1000))
    pointcloud_gt = None
    params_est = solver.solve(wrapper, pointcloud_gt, keypoints_gt)

    shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

    # print('----------------------------------------------------------------------')
    # print('ground truth parameters')
    # print('pose pca coefficients:', pose_pca)
    # print('pose global rotation:', pose_glb)
    # print('shape: pca coefficients:', shape)

    print('----------------------------------------------------------------------')
    print('estimated parameters')
    print('pose pca coefficients:', pose_pca_est)
    print('pose global rotation:', pose_glb_est)
    print('shape: pca coefficients:', shape_est)

    # mesh.set_params(pose_pca=pose_pca)
    # mesh.show_obj('gt')
    # mesh.save_obj(os.path.join(config.SAVE_PATH, './gt.obj'))
    mesh.set_params(pose_pca=pose_pca_est)
    mesh.show_obj('est')
    mesh.save_obj(os.path.join(config.SAVE_PATH, './est.obj'))

    print('ground truth and estimated meshes are saved into gt.obj and est.obj')


def result_loader_test(file_path):
    from dataloader.result_loader import KinectResultLoader
    k_loader = KinectResultLoader(file_path)
    # res = k_loader.run()
    # gen = k_loader.generator()
    # print(next(gen))
    print(k_loader.select_item(1627738564.7240, 'st', exact=False))
    # print(k_loader[3])

def result_manager_test(filepath):
    import numpy as np
    from dataloader.result_loader import ResultManager
    rm = ResultManager(filepath)
    for k_np, k_img, a_np in rm.generator():
        print(np.size(k_np), np.size(k_img), np.size(a_np))

def aaa():
    from calib import msy

if __name__ == "__main__":
    # minimal_test()
    result_manager_test('./__test__/2021-07-31 21:35:50/')
    # result_loader_test('./__test__/2021-07-31 21:35:50/')
