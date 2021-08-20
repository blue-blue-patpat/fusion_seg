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
    kpts_gt, pcl_gt, origin_scale = bridge.smpl_from_kinect(kinect_skeleton[0], kinect_pcls)

    mesh = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature, scale=1)

    wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
    solver = Solver(wrapper, max_iter=int(10e7))

    mesh_init, kpts_init = wrapper.run(np.zeros(wrapper.n_params))

    # pointcloud_gt = None
    params_est = solver.solve_full(kpts_gt, pcl_gt, origin_scale, verbose=5)

    shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

    print('----------------------------------------------------------------------')
    print('estimated parameters')
    print('pose pca coefficients:', pose_pca_est)
    print('pose global rotation:', pose_glb_est)
    print('shape: pca coefficients:', shape_est)

    mesh.set_params(pose_pca=pose_pca_est)
    mesh.save_obj(os.path.join(config.SAVE_PATH, './esttm={}.obj'.format(ymdhms_time())))


if __name__ == "__main__":
    # minimal_test()
    # result_manager_test('/home/nesc525/chen/3DSVC/__test__/2021-08-05 17:21:35')
    # result_loader_test('./__test__/mkv/')

    # from kinect.kinect_mkv import extract_mkv
    # extract_mkv("/media/nesc525/perple/2021-08-09_20-28-20/kinect/sub2/tasktm=1628512119.258128.mkv", False) 

    from kinect.kinect_skeleton import extract_skeleton
    import numpy as np
    # extract_skeleton("/home/nesc525/chen/3DSVC/__test__/mkv", "master")
    p = np.load("/home/nesc525/chen/3DSVC/__test__/mkv/kinect/master/skeleton/id=162_skid=0_st=1628392680.0206017_dt=1635153973.2593017.npy")
    print(p.shape)
