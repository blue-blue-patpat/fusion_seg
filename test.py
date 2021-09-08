from kinect.kinect_mkv import extract_mkv
from dataloader.utils import clean_dir
from optitrack.optitrack_loader import parse_opti_csv


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
    from visualization.o3d_plot import OptitrackArbeStreamPlot, KinectOfflineStreamPlotCpp, KinectArbeStreamPlot

    # parse_opti_csv("/media/nesc525/perple/2021-08-30_10-32-25/optitrack/out.csv")
    # extract_mkv("/media/nesc525/perple/2021-09-08_19-19-05/kinect/sub1/out.mkv")
    
    # from visualization.o3d_plot import OptitrackArbeStreamPlot
    # from multiprocessing import Pool
    # root_path = '/media/nesc525/perple/2021-09-08_09-45-40'


    # plot = OptitrackArbeStreamPlot(root_path, [0,-1,0,10])
    # plot.show()

    # plot = KinectOfflineStreamPlotCpp(root_path, start_frame=30, write_ply=False)
    # plot.show()

    # plot = KinectArbeStreamPlot(root_path, ["master"], [0,-1,0,2])
    # plot.show()

    from kinect.kinect_skeleton import extract_skeleton
    import os
    from dataloader.result_loader import OptitrackCSVLoader
    root_path = '/media/nesc525/perple/2021-09-08_20-18-19'
    csv_file = OptitrackCSVLoader(root_path)
    if len(csv_file):
        parse_opti_csv(csv_file.file_dict["optitrack"].loc[0,"filepath"])
    mkv_path = os.path.join(root_path, "kinect", "master")
    os.system("ignoredata/kinect_files/offline_processor {}/out.mkv {}/out.json".format(mkv_path, mkv_path))
    extract_skeleton(root_path, "master")   