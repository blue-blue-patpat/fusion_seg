from pytorch3d.ops import sample_points_from_meshes
import minimal

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


    np.random.seed(20160923)
    pose_glb = np.zeros([1, 3]) # global rotation


    ########################## smpl settings ##########################
    # note that in smpl and smpl-h no pca for pose is provided
    # therefore in the model we fake an identity matrix as the pca coefficients
    # to make the code compatible

    n_pose = 23 * 3 # degrees of freedom, (n_joints - 1) * 3
    # smpl 1.0.0: 10
    # smpl 1.1.0: 300
    n_shape = 10
    # TODO: Read pose from skeleton
    pose_pca = np.random.uniform(-0.2, 0.2, size=n_pose)
    shape = np.random.normal(size=n_shape)
    mesh = KinematicModel(config.SMPL_MODEL_1_0_PATH, armatures.SMPLArmature, scale=1)

    ########################## solving example ############################

    wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
    solver = Solver(verbose=True)

    mesh_gt, keypoints_gt = \
    mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
    pointcloud_gt = Pointclouds(sample_points_from_meshes(mesh_gt, num_samples=5000))
    params_est = solver.solve(wrapper, pointcloud_gt, keypoints_gt)

    shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

    print('----------------------------------------------------------------------')
    print('ground truth parameters')
    print('pose pca coefficients:', pose_pca)
    print('pose global rotation:', pose_glb)
    print('shape: pca coefficients:', shape)

    print('----------------------------------------------------------------------')
    print('estimated parameters')
    print('pose pca coefficients:', pose_pca_est)
    print('pose global rotation:', pose_glb_est)
    print('shape: pca coefficients:', shape_est)

    mesh.set_params(pose_pca=pose_pca)
    mesh.show_obj('gt')
    mesh.save_obj(os.path.join(config.SAVE_PATH, './gt.obj'))
    mesh.set_params(pose_pca=pose_pca_est)
    mesh.show_obj('est')
    mesh.save_obj(os.path.join(config.SAVE_PATH, './est.obj'))

    print('ground truth and estimated meshes are saved into gt.obj and est.obj')


if __name__ == "__main__":
    minimal_test()
    # from minimal import prepare_model
    # prepare_model.prepare_smpl_model()