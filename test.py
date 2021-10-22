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
    # root_path = "/home/nesc525/drivers/2/2021-10-20_14-06-35"

    # plot = OptitrackArbeStreamPlot(root_path, [0,-1,0,10])
    # plot.show()

    # plot = KinectOfflineStreamPlotCpp(root_path, start_frame=150, write_ply=False)
    # plot.show()

    # plot = KinectArbeStreamPlot(root_path, ["master"], [0,-1,0,2])
    # plot.show()

    # from kinect.kinect_skeleton import extract_skeleton
    import os
    # extract_skeleton(root_path, "master")
    # from dataloader.result_loader import OptitrackCSVLoader
    root_path = "/home/nesc525/drivers/2"
    # csv_file = OptitrackCSVLoader(root_path)
    # if len(csv_file):
    #     parse_opti_csv(csv_file.file_dict["optitrack"].loc[0,"filepath"])

    # for dev in ["sub2"]:
    #     mkv_path = os.path.join(root_path, "kinect", dev)
    #     os.system("ignoredata/kinect_files/offline_processor {}/out.mkv {}/out.json".format(mkv_path, mkv_path))
    #     extract_skeleton(root_path, dev)
    #     extract_mkv(root_path+"/kinect/{}/out.mkv".format(dev))

    # from nn.dataset import MMBody3D
    # import torch
    # dataset = MMBody3D(root_path)

    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True, pin_memory=True)
    # print(data_loader)

    # import numpy as np
    # arr = np.load("/home/nesc525/chen/3DSVC/__test__/default/arbe/id=92_st=1632382661.6592233_dt=1632382661.6915216.npy")
    # print(max(arr[:,8]))

    from run_postprocess import postprocess
    for p in os.listdir(root_path):
        if p[-1] == 'T':
            postprocess(os.path.join(root_path, p))

    # from nn.p4t.datasets.mmbody import MMBody3D
    # import pickle
    # import numpy as np
    # trans_dict = np.load("/home/nesc525/drivers/2/2021-10-20_14-06-35/calib/optitrack/optitrack_to_radar.npz")
    # print(trans_dict["t"])
    # R = trans_dict["R"]
    # t = np.asarray([-0.10785812,  3.34839662, -1.00739306])
    # np.savez("/home/nesc525/drivers/2/2021-10-20_14-06-35/calib/optitrack/optitrack_to_radar.npz", R=R, t=t)

    # dataset_all = MMBody3D(
    #         root_path=root_path,
    #         frames_per_clip=1,
    #         step_between_clips=1,
    #         num_points=1024,
    #         train=False
    # )
    # train_data = []
    # label_data = []
    # for clip, label, _ in dataset_all:
    #     train_data.append(clip)
    #     label_data.append(label)
    #     if True in np.isnan(label):
    #         print()
    # with open("ignoredata/p4t/data/test/X_data", 'wb') as f:
    #     pickle.dump(train_data, f)
    # with open("ignoredata/p4t/data/test/y_data", 'wb') as f:
    #     pickle.dump(label_data, f)

    # rmse_list = []
    # with open("ignoredata/paconv/test_result/test_RMSE-woF.txt", 'r') as f:
    #     while True:
    #         t = f.readline()
    #         if t:
    #             rmse_list.append(float(t))
    #         else:
    #             break
    # np.save("ignoredata/paconv/test_result/test_RMSE-woF", np.asarray(rmse_list))

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.interpolate import make_interp_spline

    # def plot_rmse(arr, color, label):
    #     hist, bin_edges = np.histogram(arr, bins=10, range=[0,0.2])
    #     cdf = np.cumsum(hist)/arr.size
    #     model = make_interp_spline(bin_edges[:-1], cdf, bc_type="natural")
    #     xs = np.linspace(0, 0.2, 100)
    #     ys = model(xs)
    #     plt.plot(xs, ys, color=color, label=label)

    # p4t_rmse_rs_01 = np.load("ignoredata/p4t/r_s=0.1/rmse.npy")[:,2]
    # p4t_rmse_rs_05 = np.load("ignoredata/p4t/r_s=0.5/rmse.npy")[:,2]
    # p4t_rmse_rs_07 = np.load("ignoredata/p4t/r_s=0.7/rmse.npy")[:,2]
    # p4t_rmse_rs_09 = np.load("ignoredata/p4t/r_s=0.9/rmse.npy")[:,2]
    # p4t_rmse_rs_13 = np.load("ignoredata/p4t/r_s=1.3/rmse.npy")[:,2]
    # p4t_rmse_cl_01 = np.load("ignoredata/p4t/c_l=1/rmse.npy")[:,2]
    # p4t_rmse_rt_00 = np.load("ignoredata/p4t/r_t=0/rmse.npy")[:,2]
    # p4t_rmse_rt_02 = np.load("ignoredata/p4t/r_t=2/rmse.npy")[:,2]
    # paconv_rmse_wF3 = np.load("ignoredata/paconv/test_result/test_RMSE-wF3.npy")
    # paconv_rmse_wF6 = np.load("ignoredata/paconv/test_result/test_RMSE-wF6.npy")
    # paconv_rmse_woF = np.load("ignoredata/paconv/test_result/test_RMSE-woF.npy")
    # plstm_rmse = np.load("/home/nesc525/chen/3DSVC/ignoredata/plstm/loss_plstm.npy")

    # plot_rmse(p4t_rmse_rs_01, 'red', 'r_s=0.1')
    # plot_rmse(p4t_rmse_rs_05, 'orange', 'r_s=0.5')
    # plot_rmse(p4t_rmse_rs_07, 'yellow', 'r_s=0.7')
    # plot_rmse(p4t_rmse_rs_09, 'green', 'r_s=0.9')
    # plot_rmse(p4t_rmse_rs_13, 'indigo', 'r_s=1.3')
    # plot_rmse(p4t_rmse_cl_01, 'blue', 'c_l=1')
    # plot_rmse(p4t_rmse_rt_00, 'violet', 'r_t=0')
    # plot_rmse(p4t_rmse_rt_02, 'pink', 'r_t=2')
    # # plot_rmse(paconv_rmse_wF3, 'cyan', 'f=3')
    # # plot_rmse(paconv_rmse_wF6, 'black', 'f=6')
    # # plot_rmse(paconv_rmse_woF, 'grey', 'f=0')
    # # plot_rmse(plstm_rmse, 'gold', 'plstm')

    # plt.xlabel("error (m)")
    # plt.ylabel("percentage")
    # plt.legend(loc="lower right")
    # plt.show()
    