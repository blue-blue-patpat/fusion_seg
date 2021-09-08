import os
from typing import Generator
import argparse
import numpy as np

from minimal.solver import Solver
from minimal import armatures
from minimal.models import KinematicModel, KinematicPCAWrapper
import minimal.config as config
from minimal.bridge import JointsBridge
from dataloader.result_loader import ResultFileLoader
from dataloader.utils import ymdhms_time, clean_dir
from visualization.utils import o3d_plot, o3d_coord, o3d_mesh, o3d_pcl, o3d_skeleton


def singel_minimal(jnts, pcl, save_path, verbose=-1, **kwargs):
    mesh = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

    wrapper = KinematicPCAWrapper(mesh)
    solver = Solver(wrapper, max_iter=10e7)

    if kwargs.get("show_init", False):
        mesh_init, kpts_init = wrapper.run(np.zeros(wrapper.n_params))

        o3d_plot([o3d_pcl(jnts, [0,0,1]), o3d_pcl(pcl, [1,0,0]), o3d_pcl(kpts_init, [0,1,0])], 'Minimal Input')

    params_est = solver.solve(jnts, pcl, verbose=int(verbose), mse_threshold=1e-5)

    shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

    print('----------------------------------------------------------------------')
    print('estimated parameters')
    print('pose pca coefficients:', pose_pca_est)
    print('pose global rotation:', pose_glb_est)
    print('shape: pca coefficients:', shape_est)

    mesh.set_params(pose_pca=pose_pca_est)
    solver.save_model(os.path.join(save_path, ymdhms_time()+".obj"))


def optitrack_input(root_path, **kwargs):
    loader = ResultFileLoader(root_path, int(kwargs.get("skip_head", 0)), int(kwargs.get("skip_tail", 0)),
                              enabled_sources=["arbe", "optitrack", "master", "sub1", "sub2", "kinect_pcl"])
    print(loader)
    return loader


def kinect_input(root_path, **kwargs):
    loader = ResultFileLoader(root_path, int(kwargs.get("skip_head", 0)), int(kwargs.get("skip_tail", 0)),
                              enabled_sources=["arbe", "master", "sub1", "sub2", "kinect_pcl", "kinect_skeleton"])
    print(loader)
    return loader


def optitrack_singel_frame(**kwargs):
    loader = optitrack_input(**kwargs)

    result, info = loader[kwargs.get("id", 100)]
    opti_skeleton = result["optitrack"]
    kinect_pcls = np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]])

    bridge = JointsBridge()
    bridge.init_input(opti_skeleton, kinect_pcls)
    jnts, pcl = bridge.smpl_from_optitrack()
    singel_minimal(jnts, pcl, **kwargs)


def kinect_singel_frame(**kwargs):
    loader = kinect_input(**kwargs)

    result, info = loader[kwargs.get("id", 100)]
    print(info)
    kinect_skeleton = result["master_skeleton"]
    kinect_pcls = np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]])

    bridge = JointsBridge()
    bridge.init_input(kinect_skeleton, kinect_pcls)
    jnts, pcl = bridge.map("kinect")
    singel_minimal(jnts, pcl, **kwargs)


def optitrack_stream(**kwargs):
    loader = optitrack_input(**kwargs)

    for i in range(len(loader)):
        result, info = loader[i]
        return result["optitrack"], result["master_pcl"], "id={}".format(info["arbe"]["id"])


def kinect_stream(**kwargs):
    loader = kinect_input(**kwargs)

    for i in range(len(loader)):
        result, info = loader[i]
        return result["master_skeleton"], result["master_pcl"], "id={}".format(info["arbe"]["id"])


def stream_minimal(root_path: str, verbose: int=0, **kwargs):
    save_path = os.path.join(root_path, "minimal")

    mesh = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

    wrapper = KinematicPCAWrapper(mesh)
    solver = Solver(wrapper, max_iter=10e7)

    # save to $root_path$/minimal/(param,obj,trans)
    clean_dir(save_path + "param")
    clean_dir(save_path + "obj")
    clean_dir(save_path + "trans")

    jnts_brg = JointsBridge()

    if "optitrack" in kwargs.get("task"):
        stream_source = optitrack_stream(**kwargs)
    elif "kinect" in kwargs.get("task"):
        stream_source = kinect_stream(**kwargs)
    else:
        stream_source = None

    T_skeleton, T_pcl, filename = next(stream_source)
    _jnts, _pcl = jnts_brg.map(T_skeleton, T_pcl, )

    # solve init shape
    solver.solve(_jnts, _pcl, "full", verbose=int(verbose))

    solver.save_param(os.path.join(save_path, "param", filename))
    solver.save_model(os.path.join(save_path, "obj", filename+".obj"))
    jnts_brg.save_revert_transform(os.path.join(save_path, "trans", filename))

    for skeleton, pcl, filename in stream_source:
        _jnts, _pcl = jnts_brg.map(skeleton, pcl)

        solver.solve(_pcl, _jnts, "pose", verbose=int(verbose))
        
        solver.save_param(os.path.join(save_path, "param", filename))
        solver.save_model(os.path.join(save_path, "obj", filename+".obj"))
        jnts_brg.save_revert_transform(os.path.join(save_path, "trans", filename))


def run():
    task_dict = dict(
        null=exit,
        kinect_singel_minimal=kinect_singel_frame,
        optitrack_single_minimal=optitrack_singel_frame,
        kinect_stream_minimal=stream_minimal,
        optitrack_stream_minimal=stream_minimal
    )
    parser = argparse.ArgumentParser(usage='"run_minimal.py -h" to show help.')
    parser.add_argument('-p', '--path', dest='root_path', type=str, help='File Root Path, default "./__test__/default"')
    parser.add_argument('-t', '--task', dest='task', type=str,
                        choices=list(task_dict.keys()), default='null', help='Run Target, default "null". {}'.format(task_dict))
    parser.add_argument('-a', '--addition', dest='addition', type=str,
                        default='', help='Addition args split by "#", default ""')
    args = parser.parse_args()

    args = parser.parse_args()  
    args_dict = dict([arg.split('=') for arg in args.addition.split('#') if '=' in arg])
    args_dict.update(dict(args._get_kwargs()))
    task_dict[args.task](**args_dict)



if __name__ == "__main__":
    run()
