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


def optitrack_single_frame(**kwargs):
    loader = optitrack_input(**kwargs)

    result, info = loader[kwargs.get("id", 0)]
    print(info)
    opti_skeleton = result["optitrack"]
    kinect_pcls = np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]])

    o3d_plot([o3d_pcl(kinect_pcls), o3d_pcl(opti_skeleton, [1,0,0])])
    bridge = JointsBridge()
    bridge.init_input(opti_skeleton, kinect_pcls)
    jnts, pcl = bridge.map("optitrack")
    singel_minimal(jnts, pcl, scale=bridge.scale, **kwargs)


def kinect_single_frame(**kwargs):
    loader = kinect_input(**kwargs)

    result, info = loader[kwargs.get("id", 0)]
    print(info)
    kinect_skeleton = result["master_skeleton"]
    # kinect_pcls = np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]])
    kinect_pcls = result["master_pcl"]

    o3d_plot([o3d_pcl(kinect_pcls), o3d_pcl(kinect_skeleton, [1,0,0])])
    bridge = JointsBridge()
    bridge.init_input(kinect_skeleton, kinect_pcls)
    jnts, pcl = bridge.map("kinect")
    singel_minimal(jnts, pcl, scale=bridge.scale, **kwargs)


def optitrack_stream(**kwargs):
    loader = optitrack_input(**kwargs)

    for i in range(len(loader)):
        result, info = loader[i]
        yield result["optitrack"], result["master_pcl"], "id={}".format(info["arbe"]["id"])


def kinect_stream(**kwargs):
    loader = kinect_input(**kwargs)

    for i in range(len(loader)):
        result, info = loader[i]
        yield result["master_skeleton"], result["master_pcl"], "id={}".format(info["arbe"]["id"])


def singel_minimal(jnts, pcl, save_path, scale=1, dbg_level=-1, plot_type="open3d", **kwargs):
    dbg_level = int(dbg_level)

    mesh = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

    wrapper = KinematicPCAWrapper(mesh)
    solver = Solver(wrapper, max_iter=40, plot_type=plot_type)

    if dbg_level > -1:
        mesh_init, kpts_init = wrapper.run(np.zeros(wrapper.n_params))

        o3d_plot([o3d_pcl(jnts, [0,0,1]), o3d_pcl(pcl, [1,0,0]), o3d_pcl(kpts_init, [0,1,0])], 'Minimal Input')

    params_est = solver.solve(jnts, pcl, dbg_level=dbg_level, mse_threshold=1e-4, kpts_threshold=0.02/scale)

    shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

    print('----------------------------------------------------------------------')
    print('estimated parameters')
    print('pose pca coefficients:', pose_pca_est)
    print('pose global rotation:', pose_glb_est)
    print('shape: pca coefficients:', shape_est)

    mesh.set_params(pose_pca=pose_pca_est)
    solver.save_model(os.path.join(save_path, ymdhms_time()+".obj"))


def stream_minimal(root_path: str, dbg_level: int=0, plot_type="open3d", **kwargs):
    save_path = os.path.join(root_path, "minimal")
    dbg_level = int(dbg_level)

    smpl = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

    wrapper = KinematicPCAWrapper(smpl)
    solver = Solver(wrapper, plot_type=plot_type)

    # save to $root_path$/minimal/(param,obj,trans)
    clean_dir(os.path.join(save_path, "param"))
    clean_dir(os.path.join(save_path, "obj"))
    clean_dir(os.path.join(save_path, "trans"))
    clean_dir(os.path.join(save_path, "loss"))

    jnts_brg = JointsBridge()

    if "optitrack" in kwargs.get("task"):
        stream_source = optitrack_stream(root_path=root_path, **kwargs)
        data_type = "optitrack"
    elif "kinect" in kwargs.get("task"):
        stream_source = kinect_stream(root_path=root_path, **kwargs)
        data_type = "kinect"
    else:
        raise NotImplementedError()

    T_skeleton, T_pcl, filename = next(stream_source)
    jnts_brg.init_input(T_skeleton, T_pcl)
    _jnts, _pcl = jnts_brg.map(data_type)

    # solve init shape
    _, losses = solver.solve(_jnts, _pcl, "full", dbg_level=dbg_level, max_iter=100)

    solver.save_param(os.path.join(save_path, "param", filename))
    solver.save_model(os.path.join(save_path, "obj", filename+".obj"))
    jnts_brg.save_revert_transform(os.path.join(save_path, "trans", filename))
    losses.save_losses(os.path.join(save_path, "loss", filename))
    del losses

    # disable mesh update
    solver.model.core.compute_mesh = False

    # init pose update losses
    losses_w = dict(
        kpts_losses=1,
        edge_losses=0,
        face_losses=0,
    )

    for skeleton, pcl, filename in stream_source:
        jnts_brg.init_input(skeleton, pcl)
        _jnts, _pcl = jnts_brg.map(data_type)

        _, losses = solver.solve(_jnts, _pcl, "pose", max_iter=80, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)
        
        solver.save_param(os.path.join(save_path, "param", filename))
        solver.save_model(os.path.join(save_path, "obj", filename+".obj"))
        jnts_brg.save_revert_transform(os.path.join(save_path, "trans", filename))
        losses.save_losses(os.path.join(save_path, "loss", filename))
        del losses


def optitrack_stream_windowed_minimal(root_path: str, dbg_level: int=0, window_len: int=2, plot_type="open3d", **kwargs):
    save_path = os.path.join(root_path, "minimal")
    temp_path = os.path.join(root_path, "minimal_temp")
    dbg_level = int(dbg_level)
    window_len = int(window_len)

    smpl = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

    wrapper = KinematicPCAWrapper(smpl)
    solver = Solver(wrapper, plot_type=plot_type)

    # save to $root_path$/minimal/(param,obj,trans)
    clean_dir(os.path.join(save_path, "param"))
    clean_dir(os.path.join(save_path, "obj"))
    clean_dir(os.path.join(save_path, "trans"))
    clean_dir(os.path.join(save_path, "loss"))

    clean_dir(os.path.join(temp_path, "param"))
    clean_dir(os.path.join(temp_path, "obj"))
    clean_dir(os.path.join(temp_path, "trans"))
    clean_dir(os.path.join(temp_path, "loss"))

    jnts_brg = JointsBridge()

    loader = optitrack_input(root_path, **kwargs)

    data_type = "optitrack"

    if os.path.exists(os.path.join(save_path, "init_params.npz")):
        # load init shape & pose
        solver.update_params(np.load(os.path.join(save_path, "init_params.npz")))
    else:
        # solve init shape
        shape_params = []
        for i in range(window_len*2+1):
            result, info = loader[i]
            jnts_brg.init_input(result["optitrack"], np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]]))
            _jnts, _pcl = jnts_brg.map(data_type)

            _, losses = solver.solve(_jnts, _pcl, "full", dbg_level=dbg_level, max_iter=100)
            
            shape_params.append(solver.shape_params)
            del losses
    
        solver.shape_params = np.array(shape_params).mean(0)
        solver.save_param(os.path.join(save_path, "init_params"))

    # disable mesh update
    # solver.model.core.compute_mesh = False

    # init pose update losses
    losses_w = dict(
        kpts_losses=1,
        edge_losses=50,
        face_losses=50,
    )

    inputs = MinimalInput(loader, jnts_brg, data_type)

    for i in range(len(loader)):
        init_pose = solver.pose_params
        results = {}
        for j in range(max(0, i-window_len), min(len(loader), i+window_len)):
            solver.update_params(init_pose)
            _, losses = solver.solve(inputs[j]["jnts"], inputs[i]["pcl"], "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)
        
            filename = "id={}#{}_rid={}".format(i, j, inputs[i]["info"]["arbe"]["id"])
            solver.save_param(os.path.join(temp_path, "param", filename))
            solver.save_model(os.path.join(temp_path, "obj", filename+".obj"))
            inputs.save_revert_transform(j, os.path.join(temp_path, "trans", filename))
            losses.save_losses(os.path.join(temp_path, "loss", filename))
            results[j] = dict(
                pose = solver.pose_params,
                loss = losses[-1]
            )
            del losses
        result_key = min(results, key=lambda key: results[key]["loss"])
        solver.update_params(results[result_key]["pose"])
        filename = "id={}#{}_rid={}".format(i, j, inputs[i]["info"]["arbe"]["id"])

        solver.save_param(os.path.join(save_path, "param", filename))
        solver.save_model(os.path.join(save_path, "obj", filename+".obj"))
        inputs.save_revert_transform(j, os.path.join(save_path, "trans", filename))
        inputs.remove(i-window_len)


class MinimalInput:
    def __init__(self, loader: ResultFileLoader, brg: JointsBridge, data_type: str) -> None:
        self.input_dict = {}
        self.loader = loader
        self.brg = brg
        self.data_type = data_type

    def update(self, idx: int):
        if idx in self.input_dict.keys():
            return
        result, info = self.loader[idx]
        self.brg.init_input(result["optitrack"], np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]]))
        _jnts, _pcl = self.brg.map(self.data_type)
        R, t, scale = self.brg.revert_transform()
        self.input_dict[idx] = dict(
            jnts=_jnts,
            pcl=_pcl,
            info=info,
            transform=dict(
                R=R, t=t, scale=scale
            )
        )

    def remove(self, idx: int):
        if idx in self.input_dict.keys():
            del self.input_dict[idx]

    def save_revert_transform(self, idx: int, file_path: str):
        np.savez(file_path, **self.input_dict[idx]["transform"])

    def __getitem__(self, idx):
        self.update(idx)
        return self.input_dict[idx]


def run():
    task_dict = dict(
        null=exit,
        kinect_single_minimal=kinect_single_frame,
        optitrack_single_minimal=optitrack_single_frame,
        kinect_stream_minimal=stream_minimal,
        optitrack_stream_minimal=stream_minimal,
        optitrack_stream_windowed=optitrack_stream_windowed_minimal,
    )
    parser = argparse.ArgumentParser(usage='"run_minimal.py -h" to show help.')
    parser.add_argument('-p', '--path', dest='root_path', type=str, help='File Root Path, default "./__test__/default"')
    parser.add_argument('-t', '--task', dest='task', type=str,
                        choices=list(task_dict.keys()), default='null', help='Run Target, default "null". {}'.format(task_dict))
    parser.add_argument('-a', '--addition', dest='addition', type=str,
                        default='', help='Addition args split by "#", default ""')
    args = parser.parse_args()

    args_dict = dict([arg.split('=') for arg in args.addition.split('#') if '=' in arg])
    args_dict.update(dict(args._get_kwargs()))
    task_dict[args.task](**args_dict)

if __name__ == "__main__":
    run()
