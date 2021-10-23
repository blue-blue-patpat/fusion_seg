import os
import gc
import json
import argparse
from time import sleep
import numpy as np
import torch

from minimal import armatures
import minimal.config as config
from minimal.bridge import JointsBridge
from minimal.input_loader import KINECT_SUB1_SOURCE, OPTI_DATA, KINECT_DATA, MinimalInput
from minimal.utils import get_freer_gpu
from dataloader.result_loader import MinimalLoader, ResultFileLoader
from dataloader.utils import ymdhms_time, clean_dir, create_dir
from message.dingtalk import MSG_ERROR, MSG_INFO, TimerBot


def optitrack_input(root_path, **kwargs):
    loader = ResultFileLoader(root_path, int(kwargs.get("skip_head", 0)), int(kwargs.get("skip_tail", 0)),
                              enabled_sources=["optitrack", "master", "sub1", "sub2", "kinect_pcl", "kinect_pcl_remove_zeros"])
    print(loader)
    return loader


def kinect_input(root_path, **kwargs):
    loader = ResultFileLoader(root_path, int(kwargs.get("skip_head", 0)), int(kwargs.get("skip_tail", 0)),
                              enabled_sources=["master", "sub1", "sub2", "kinect_pcl", "kinect_pcl_remove_zeros", "kinect_skeleton"])
    print(loader)
    return loader


def optitrack_single_frame(**kwargs):
    from visualization.utils import o3d_plot, o3d_pcl

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
    from visualization.utils import o3d_plot, o3d_pcl

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
    from visualization.utils import o3d_plot, o3d_pcl
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.solver import Solver

    dbg_level = int(dbg_level)

    mesh = KinematicModel().init_from_file(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

    wrapper = KinematicPCAWrapper(mesh)
    solver = Solver(wrapper, plot_type=plot_type)

    if dbg_level > -1:
        mesh_init, kpts_init = wrapper.run(np.zeros(wrapper.n_params + 3))

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
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.solver import Solver

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


def stream_windowed_minimal(root_path: str, dbg_level: int=0, window_len: int=2, plot_type="open3d", device="cpu", data_type=OPTI_DATA, switch_skel_loss_threshold=1.0, **kwargs):
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.solver import Solver
    from minimal.models_torch import KinematicModel as KinematicModelTorch, KinematicPCAWrapper as KinematicPCAWrapperTorch
    from minimal.solver_torch import Solver as SolverTorch
    
    bot = kwargs.pop("msg_bot")
    assert isinstance(bot, TimerBot)
    bot.enable()
    
    bot.print("{} : [Minimal] Starting minimal...\nroot_path={}\tdbg_level={}\twindow_len={}".format(
        ymdhms_time(), root_path, dbg_level, window_len
    ))

    save_path = os.path.join(root_path, "minimal")
    dbg_level = int(dbg_level)
    window_len = int(window_len)

    if device == "cpu":
        bot.print("{} : [Minimal] Running in CPU mode.".format(ymdhms_time()))
        _device = None
        _Model = KinematicModel
        _Wrapper = KinematicPCAWrapper
        _Solver = Solver
    else:
        _device = torch.device("cuda:0")
        bot.print("{} : [Minimal] Running in GPU mode, {}".format(ymdhms_time(), device))
        _Model = KinematicModelTorch
        _Wrapper = KinematicPCAWrapperTorch
        _Solver = SolverTorch

    record_info_path = os.path.join(root_path, "infomation.json")

    # female model by default
    model_path = config.SMPL_MODEL_1_0_PATH
    if os.path.exists(record_info_path):
        with open(record_info_path, "r") as f:
            record_info = json.load(f)
            if record_info.get(" gender", None) == "male" or record_info.get("gender", None) == "male":
                model_path = config.SMPL_MODEL_1_0_MALE_PATH

    smpl = _Model(device=_device).init_from_file(model_path, armatures.SMPLArmature)
    wrapper = _Wrapper(smpl)
    solver = _Solver(wrapper, plot_type=plot_type)

    # save to $root_path$/minimal/(param,obj,trans)
    create_dir(os.path.join(save_path, "param"))
    create_dir(os.path.join(save_path, "obj"))
    # create_dir(os.path.join(save_path, "trans"))
    # create_dir(os.path.join(save_path, "loss"))

    jnts_brg = JointsBridge()

    if data_type == OPTI_DATA:
        loader = optitrack_input(root_path, **kwargs)
    elif data_type == KINECT_DATA:
        loader = kinect_input(root_path, **kwargs)
    else:
        raise NotImplementedError
    
    bot.print("{} : [Minimal] Running {} fitting.".format(ymdhms_time(), data_type))

    losses_w = dict(
        kpts_losses=1,
        edge_losses=5000,
        face_losses=5000,
    )

    if os.path.exists(os.path.join(save_path, "init_params.npz")):
        # load init shape & pose
        bot.print("{} : [Minimal] Load current init params".format(ymdhms_time()))
        solver.update_params(np.load(os.path.join(save_path, "init_params.npz")))
        # jnts_brg.set_scale(np.load(os.path.join(save_path, "init_transform.npz"))["scale"])
    else:
        # solve init shape
        bot.print("{} : [Minimal] Start solving init params...".format(ymdhms_time()))
        shape_params = []
        for i in range(window_len*2+1):
            result, info = loader[i]
            if data_type == OPTI_DATA:
                jnts_brg.init_input(result["optitrack"], np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]]))
            elif data_type == KINECT_DATA:
                # input is mean of sub1 & sub2
                jnts_brg.init_input(np.mean([result["sub1_skeleton"], result["sub2_skeleton"]], axis=0), np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]]))
            else:
                raise NotImplementedError

            _jnts, _pcl = jnts_brg.map(data_type)

            init_pose = np.zeros(solver.model.n_pose + solver.model.n_coord + solver.model.n_glb)
            # translation
            init_pose[:3] = -(_jnts.max(axis=0) + _jnts.min(axis=0))/2
            # rotation
            init_pose[3] = 0.5
            solver.update_params(init_pose)

            _, losses = solver.solve(_jnts, _pcl, "full", dbg_level=dbg_level, max_iter=100, losses_with_weights=losses_w)
            
            shape_params.append(solver.shape_params)
            del losses
    
        if device == "cpu":
            solver.shape_params = np.array(shape_params).mean(0)
        else:
            solver.shape_params = torch.vstack(shape_params).mean(0)
        solver.save_param(os.path.join(save_path, "init_params"))
        # jnts_brg.save_revert_transform(os.path.join(save_path, "init_transform"))

    # disable mesh update
    # solver.model.core.compute_mesh = False

    # init pose update losses
    losses_w = dict(
        kpts_losses=1,
        edge_losses=50,
        face_losses=50,
    )

    bot.print("{} : [Minimal] Losses: {}".format(ymdhms_time(), losses_w))

    inputs = MinimalInput(loader, jnts_brg.scale, data_type)

    # set init skeleton source as sub1
    if data_type == KINECT_DATA:
        inputs.jnts_source = KINECT_SUB1_SOURCE

    current_minimal = MinimalLoader(root_path)

    start_idx = 0

    for i in range(len(loader)):
        result = current_minimal.select_item_in_tag(i+int(kwargs.get("skip_head", 0)), "rid", "minimal/param")
        if len(result) == 0:
            break
        start_idx = i

    bot.print("{} : [Minimal] Configure start index {}".format(ymdhms_time(), start_idx))

    for i in range(start_idx, len(loader)):
        rid = i+int(kwargs.get("skip_head", 0))
        init_pose = solver.pose_params
        empty_pcl = np.array([[0,0,0]])
        results = {}
        for j in range(max(0, i-window_len), min(len(loader), i+window_len)):
            solver.update_params(init_pose)
            pcl_source = inputs[i]
            _, losses = solver.solve(inputs[j]["jnts"], pcl_source["pcl"] if "pcl" in pcl_source.keys() else empty_pcl, "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)

            results[j] = dict(
                pose = solver.pose_params,
                loss = losses[-1]
            )

            # if loss exceeds threshold, try switching to another skeleton source
            if data_type == KINECT_DATA and losses[-1] > switch_skel_loss_threshold:
                _, new_losses = solver.solve(inputs[j]["jnts"], pcl_source["pcl"] if "pcl" in pcl_source.keys() else empty_pcl, "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)
                if new_losses[-1] < losses[-1]:
                    results[j] = dict(
                        pose = solver.pose_params,
                        loss = new_losses[-1]
                    )
                    # switch to skeleton source of another device
                    inputs.jnts_source = 1 - inputs.jnts_source
                del new_losses

            del losses
        result_key = min(results, key=lambda key: results[key]["loss"])
        solver.update_params(results[result_key]["pose"])
        if inputs[j]["info"].get("nan", False):
            nan_flag = "nan"
        else:
            nan_flag = "fine"
        filename = "id={}_skid={}_masid={}_rid={}_type={}".format(i, inputs[j]["info"]["optitrack"]["id"], inputs[i]["info"]["master_pcl"]["id"], rid, nan_flag)

        solver.save_param(os.path.join(save_path, "param", filename))
        solver.save_model(os.path.join(save_path, "obj", filename+".obj"))
        # inputs.save_revert_transform(j, os.path.join(save_path, "trans", filename))
        inputs.remove(i-window_len)

        bot.print("{} : [Minimal] {} Frame rid={} with loss {:.4}".format(ymdhms_time(), root_path[-21:-2], rid, results[result_key]["loss"]))
        gc.collect()
    bot.add_task("{} : [Minimal] {} finished.".format(ymdhms_time(), root_path[-21:-2]), MSG_ERROR)
    sleep(15)


def check(root_path, **kwargs):
    from visualization.mesh_plot import MinimalResultStreamPlot

    MinimalResultStreamPlot(root_path, skip_head=kwargs.get("skip_head", 0), skip_tail=kwargs.get("skip_tail", 0)).show()


def run():
    task_dict = dict(
        null=exit,
        kinect_single_minimal=kinect_single_frame,
        optitrack_single_minimal=optitrack_single_frame,
        kinect_stream_minimal=stream_minimal,
        optitrack_stream_minimal=stream_minimal,
        stream_windowed=stream_windowed_minimal,
        check=check,
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

    args_dict["msg_bot"] = TimerBot(args_dict.get("interval", 10))
    
    if args_dict.get("device", "cpu") == "cpu":
        print("CPU mode")
    elif args_dict.get("device", "cpu").split(":")[-1].isdigit():
        gpu_id = args_dict.get("device", "cpu").split(":")[-1]
        print("Custom GPU {}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    else:
        gpu_id = get_freer_gpu()
        print("Auto select GPU {}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        args_dict["device"] = "cuda:{}".format(gpu_id)

    try:
        task_dict[args.task](**args_dict)
    except Exception as e:
        args_dict["msg_bot"].enable()
        args_dict["msg_bot"].add_task("{} : task={}, path={}\ndetails: {}".format(ymdhms_time, args.task, args.root_path, e), MSG_ERROR)
        sleep(15)
        raise e

if __name__ == "__main__":
    run()
