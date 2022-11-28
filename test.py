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
    root_path = "/home/nesc525/drivers/3"
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

    # from run_postprocess import postprocess
    # for p in os.listdir(root_path):
    #     if p[-1] == 'A':
    #         postprocess(os.path.join(root_path, p))

    from nn.p4t.datasets.mmbody import MMBody3D
    import pickle
    import numpy as np
    p = "/home/nesc525/drivers/1/2021-10-23_21-13-05_N/calib"
    p_o = p + "/optitrack/optitrack_to_radar.npz"
    trans_dict = np.load(p_o)
    print(trans_dict["t"])
    R = trans_dict["R"]
    t = np.asarray([-0.20738628  ,2.96957179, -1.15383591])
    np.savez(p_o, R=R, t=t)

    p_k = p + "/kinect/master_to_world.npz"
    trans_dict = np.load(p_k)
    print(trans_dict["t"])
    R = trans_dict["R"]
    t = np.asarray([-0.00723176, -0.67711437,  4.76285076])
    np.savez(p_k, R=R, t=t)
    import numpy as np
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
    
    import torch
    device=torch.device('cpu')
    blank_atom=torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32, requires_grad=False, device=device)
    q=torch.rand((7, 13, 9, 3, 3), dtype=torch.float32, device='cpu')
    batch_size=q.size()[0]
    length_size=q.size()[1]
    q=q.view(batch_size*length_size, 9, 3, 3)
    q_blank=blank_atom.repeat(batch_size*length_size, 1, 1, 1)
    pose=torch.cat((q_blank,
                    q[:,1:3,:,:],
                    q_blank,
                    q[:,3:5,:,:],
                    q_blank.repeat(1,10,1,1),
                    q[:,5:9,:,:],
                    q_blank.repeat(1,4,1,1)), 1)
    rotmat=q[:,0,:,:]

    import torch
    import torch.nn as nn
    def rot_mat_2_euler(R):
        batch = R.size()[0]
        sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
        singular= sy<1e-6
        singular=singular.float()
            
        x=torch.atan2(R[:,2,1], R[:,2,2])
        y=torch.atan2(-R[:,2,0], sy)
        z=torch.atan2(R[:,1,0],R[:,0,0])
        
        xs=torch.atan2(-R[:,1,2], R[:,1,1])
        ys=torch.atan2(-R[:,2,0], sy)
        zs=R[:,1,0]*0
            
        out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
        out_euler[:,0]=x*(1-singular)+xs*singular
        out_euler[:,1]=y*(1-singular)+ys*singular
        out_euler[:,2]=z*(1-singular)+zs*singular
        
        return out_euler

    def euler_2_rot_mat(euler):

        batch=euler.shape[0]
            
        c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
        s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
        c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
        s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
        c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
        s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
            
        row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
        row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
        row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
            
        matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
        
        return matrix

    def rotation6d_2_euler(nn_output):
        batch_size = nn_output.size()[0]
        num_joints = 9
        blank_atom=torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32, requires_grad=False, device=torch.device('cuda:0'))
        q_blank=blank_atom.repeat(batch_size, 1, 1, 1)
        pose = nn_output[:,3:num_joints*6+3].reshape(batch_size*num_joints, 6).contiguous()
        tmp_x = nn.functional.normalize(pose[:,:3], dim = -1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, pose[:,3:], dim = -1), dim = -1)
        tmp_y = torch.cross(tmp_z, tmp_x, dim = -1)

        tmp_x = tmp_x.view(batch_size,num_joints, 3, 1)
        tmp_y = tmp_y.view(batch_size,num_joints, 3, 1)
        tmp_z = tmp_z.view(batch_size,num_joints, 3, 1)
        pose = torch.cat((tmp_x, tmp_y, tmp_z), -1)
        R=torch.cat((q_blank,
                    pose[:,1:3,:,:],
                    q_blank,
                    pose[:,3:5,:,:],
                    q_blank.repeat(1,10,1,1),
                    pose[:,5:9,:,:],
                    q_blank.repeat(1,4,1,1)), 1).view(batch_size*24,3,3)
        rotmat=pose[:,0,:,:]

    R = torch.rand(1, 3, 3, device='cuda:0')

    euler = torch.tensor([[1.,1,1],[2,1,1]], device='cuda:0')
    output = torch.rand(2,72, device='cuda:0')
    euler = rotation6d_2_euler(output)

    # euler = rot_mat_2_euler(R)
    mat = euler_2_rot_mat(euler)
    euler2 = rot_mat_2_euler(mat)
    print(euler == euler2)
    print(mat == R)

    
"""
vis_smpl
"""
# from visualization.utils import o3d_smpl_mesh, o3d_plot
# import pandas as pd
# import numpy as np


# # amass_pose_df = pd.read_csv("./ignoredata/visualize_files/test_pose.csv", index_col=0)
# test_pose = np.hstack((np.zeros(3), np.load('/home/nesc525/drivers/5/AMASS/CMU/16/16_35_poses.npz')["poses"][0,:21*3], np.zeros(6)))

# pose_df = pd.read_csv("./ignoredata/visualize_files/mm_pose.csv", index_col=0)

# # test_pose = np.hstack((np.zeros(6), np.asarray(amass_pose_df.iloc[0])[:-1]))
# mm_pose = np.hstack((np.zeros(6), np.asarray(pose_df.iloc[0])[:-1]))

# o3d_plot([o3d_smpl_mesh(test_pose, [0,1,0]), ])


"""
smpl_test
"""

# from torch import tensor
# from minimal.config import SMPL_MODEL_1_0_PATH, SMPL_MODLE_RAW_1_0_MALE_PATH, SMPL_MODLE_RAW_1_0_NEUTRAL_PATH
# from nn.SMPL.smpl_layer import SMPLModel
# from minimal.models_torch import KinematicModel, KinematicPCAWrapper
# import torch
# import numpy as np
# from visualization.utils import o3d_plot, o3d_mesh


# param = np.load("/home/nesc525/drivers/1/2021-10-16_17-23-57_N/minimal/param/id=1085_skid=1321_masid=1132_rid=1085_type=fine.npz")
# pose = torch.from_numpy(param["pose"]).to(torch.float64)
# shape = torch.from_numpy(param["shape"]).to(torch.float64)

# smpl = SMPLModel()
# smpl_m = SMPLModel(model_path=SMPL_MODLE_RAW_1_0_MALE_PATH)
# smpl_n = SMPLModel(model_path=SMPL_MODLE_RAW_1_0_NEUTRAL_PATH)

# _smpl = KinematicPCAWrapper(KinematicModel(torch.device("cpu")).init_from_file(SMPL_MODEL_1_0_PATH, compute_mesh=False))

# v, j = smpl(shape, pose[3:], -pose[:3])
# v_m, j = smpl_m(shape, pose[3:], -pose[:3])
# v_n, j = smpl_n(shape, pose[3:], -pose[:3])

# _smpl.run(torch.cat((pose, shape),-1))

# f = _smpl.core.faces

# o3d_plot([o3d_mesh([v+torch.tensor([1,0,0]), f], [1,0.3,0.3]), o3d_mesh([v_m-torch.tensor([1,0,0]), f], [0.3,0.3,1]), o3d_mesh([v_n, f])])


# # from run_postprocess import postprocess
# # import os


# # path = "/home/nesc525/drivers/3"
# # dir_list = dir_list = os.listdir(path)

# # for dir in dir_list:
# #     if "2021" in dir:
# #         postprocess(os.path.join(path, dir))
