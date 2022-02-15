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
from minimal.input_loader import KINECT_SUB1_SOURCE, KINECT_SUB2_SOURCE, OPTI_DATA, KINECT_DATA, MinimalInput
from minimal.utils import get_freer_gpu, get_gpu_count
from dataloader.result_loader import MinimalLoader, ResultFileLoader
from dataloader.utils import ymdhms_time, clean_dir, create_dir
from message.dingtalk import MSG_ERROR, MSG_INFO, TimerBot


def optitrack_input(root_path, **kwargs):
    loader = ResultFileLoader(root_path, int(kwargs.get("skip_head", 0)), int(kwargs.get("skip_tail", 0)),
                              enabled_sources=["optitrack", "sub1", "sub2", "kinect_pcl", "kinect_pcl_remove_zeros"])
    print(loader)
    return loader


def kinect_input(root_path, **kwargs):
    loader = ResultFileLoader(root_path, int(kwargs.get("skip_head", 0)), int(kwargs.get("skip_tail", 0)),
                              enabled_sources=["sub1", "sub2", "kinect_pcl", "kinect_pcl_remove_zeros", "kinect_skeleton"])
    print(loader)
    return loader


def optitrack_single_frame(**kwargs):
    from visualization.utils import o3d_plot, o3d_pcl

    loader = optitrack_input(**kwargs)

    result, info = loader[kwargs.get("id", 0)]
    print(info)
    opti_skeleton = result["optitrack"]
    kinect_pcls = np.vstack([result["sub1_pcl"], result["sub2_pcl"]])

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
    kinect_skeleton = result["sub1_skeleton"]
    kinect_pcls = np.vstack([result["sub1_pcl"], result["sub2_pcl"]])

    o3d_plot([o3d_pcl(kinect_pcls), o3d_pcl(kinect_skeleton, [1,0,0])])
    bridge = JointsBridge()
    bridge.init_input(kinect_skeleton, kinect_pcls)
    jnts, pcl = bridge.map("kinect")
    singel_minimal(jnts, pcl, scale=bridge.scale, **kwargs)


def parallel_single_frame(**kwargs):
    from visualization.utils import o3d_plot, o3d_pcl, o3d_mesh

    opti_loader = optitrack_input(**kwargs)
    kinect_loader = kinect_input(**kwargs)

    opti_result, _ = opti_loader[kwargs.get("id", 0)]
    kinect_result, _ = kinect_loader[kwargs.get("id", 0)]
    gender = opti_result["information"].get("gender", "female")
    # print(info)
    opti_skeleton = opti_result["optitrack"]
    kinect_pcls = np.vstack([kinect_result["sub1_pcl"], kinect_result["sub2_pcl"]])

    kinect_skeleton = kinect_result["sub1_skeleton"]

    bridge = JointsBridge()
    bridge.init_input(opti_skeleton, kinect_pcls)
    opti_jnts, opti_pcl = bridge.map()
    bridge.init_input(kinect_skeleton, kinect_pcls)
    kinect_jnts, kinect_pcl = bridge.map("kinect")

    # o3d_plot([o3d_pcl(opti_skeleton[[6,7,11,18]], [1,0,0]), o3d_pcl(kinect_pcls, [0,1,0])])
    
    o3d_plot([o3d_pcl(kinect_pcl, [0,1,0]), o3d_pcl(kinect_jnts, [1,0,0]), o3d_pcl(opti_jnts, [0,0,1])])

    opti_mesh, _ = singel_minimal(opti_jnts, opti_pcl, gender=gender, **kwargs)

    kinect_mesh, _ = singel_minimal(kinect_jnts, kinect_pcl, gender=gender, **kwargs)

    o3d_plot([o3d_mesh(opti_mesh, [1,0.3,0.3]), o3d_mesh(kinect_mesh, [0.3,1,0.3])])


def singel_minimal(jnts, pcl, save_path, scale=1, dbg_level=-1, plot_type="open3d", gender="feamle", device="cpu", **kwargs):
    from visualization.utils import o3d_plot, o3d_pcl, o3d_mesh
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.solver import Solver
    from minimal.models_torch import KinematicModel as KinematicModelTorch, KinematicPCAWrapper as KinematicPCAWrapperTorch
    from minimal.solver_torch import Solver as SolverTorch

    dbg_level = int(dbg_level)
    
    if device == "cpu":
        _device = None
        _Model = KinematicModel
        _Wrapper = KinematicPCAWrapper
        _Solver = Solver
    else:
        _device = torch.device("cuda:0")
        _Model = KinematicModelTorch
        _Wrapper = KinematicPCAWrapperTorch
        _Solver = SolverTorch
    
    smpl = _Model(device=_device).init_from_file(config.SMPL_MODEL_1_0_MALE_PATH if gender == "male" else config.SMPL_MODEL_1_0_PATH)

    wrapper = _Wrapper(smpl)
    solver = _Solver(wrapper, plot_type=plot_type)

    if dbg_level > -1:
        init_param = np.zeros(wrapper.n_params + 3)
        # translation
        init_param[:3] = -(jnts.max(axis=0) + jnts.min(axis=0))/2
        # rotation
        init_param[3] = np.pi/2

        solver.update_params(init_param)
        # o3d_plot([o3d_pcl(jnts, [0,0,1]), o3d_pcl(pcl, [1,0,0]), o3d_pcl(kpts_init, [0,1,0]), o3d_mesh(mesh_init)], 'Minimal Input')
    
    params_est, losses = solver.solve(jnts, pcl, dbg_level=dbg_level, mse_threshold=1e-4)
    print(losses[-1])

    mesh_est, keypoints_est = wrapper.run(params_est)
    # solver.save_model(os.path.join(save_path, ymdhms_time()+".obj"))
    if save_path is not None:
        solver.save_param(save_path)
    return [wrapper.core.verts.cpu().numpy(), wrapper.core.faces.cpu().numpy()], keypoints_est


def stream_kinect_optitrack_windowed(root_path: str, dbg_level: int=0, window_len: int=2, plot_type="open3d", device="cpu", switch_skel_loss_threshold=1.0, **kwargs):
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.solver import Solver
    from minimal.models_torch import KinematicModel as KinematicModelTorch, KinematicPCAWrapper as KinematicPCAWrapperTorch
    from minimal.solver_torch import Solver as SolverTorch
    # from visualization.utils import o3d_plot, o3d_pcl
    
    bot = kwargs.pop("msg_bot")
    assert isinstance(bot, TimerBot)
    # bot.enable()
    
    bot.print("{} : [Minimal] Starting minimal...\nroot_path={}\tdbg_level={}\twindow_len={}".format(
        ymdhms_time(), root_path, dbg_level, window_len
    ))

    opti_save_path = os.path.join(root_path, "minimal")
    kinect_save_path = os.path.join(root_path, "minimal_k")
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

    smpl = _Model(device=_device).init_from_file(model_path)
    opti_solver = _Solver(_Wrapper(smpl), plot_type=plot_type)
    kinect_solver = _Solver(_Wrapper(_Model().init_from_model(smpl)), plot_type=plot_type)

    # save to $root_path$/minimal/(param,obj,trans)
    create_dir(os.path.join(opti_save_path, "param"))
    create_dir(os.path.join(opti_save_path, "obj"))
    create_dir(os.path.join(kinect_save_path, "param"))
    create_dir(os.path.join(kinect_save_path, "obj"))

    jnts_brg = JointsBridge()

    opti_loader = optitrack_input(root_path, **kwargs)
    kinect_loader = kinect_input(root_path, **kwargs)
    
    bot.print("{} : [Minimal] Running {} fitting.".format(ymdhms_time(), "Parallel"))

    losses_w = dict(
        kpts_losses=1,
        edge_losses=50,
        face_losses=50,
        vposer_j_loss=1,
        vposer_v_loss=1,
        vposer_p_loss=1
    )

    # Opti
    if os.path.exists(os.path.join(opti_save_path, "init_params.npz")):
        # load init shape & pose
        bot.print("{} : [Minimal] Load current init params".format(ymdhms_time()))
        opti_solver.update_params(np.load(os.path.join(opti_save_path, "init_params.npz")))
    else:
        # solve init shape
        bot.print("{} : [Minimal] Start solving init params...".format(ymdhms_time()))
        opti_shape_params = []

        for i in range(window_len*2+1):
            result, info = opti_loader[i]
            jnts_brg.init_input(result["optitrack"], np.vstack([result["sub1_pcl"], result["sub2_pcl"]]))

            _jnts, _pcl = jnts_brg.map(OPTI_DATA)

            # o3d_plot([o3d_pcl(_jnts, [1,0,0]), o3d_pcl(result["optitrack"], [0,0,1]), o3d_pcl(_pcl, [0,1,0])])

            init_param = np.zeros(opti_solver.model.n_pose + opti_solver.model.n_coord + opti_solver.model.n_glb + opti_solver.model.n_shape)
            # translation
            init_param[:3] = -(_jnts.max(axis=0) + _jnts.min(axis=0))/2
            # rotation
            init_param[3] = np.pi/2
            opti_solver.update_params(init_param)

            _, losses = opti_solver.solve(_jnts, _pcl, "full", dbg_level=dbg_level, max_iter=60, losses_with_weights=losses_w)
            
            opti_shape_params.append(opti_solver.shape_params)
            del losses
    
        if device == "cpu":
            opti_solver.shape_params = np.array(opti_shape_params).mean(0)
        else:
            opti_solver.shape_params = torch.vstack(opti_shape_params).mean(0)
        opti_solver.save_param(os.path.join(opti_save_path, "init_params"))

    # Kinect
    if os.path.exists(os.path.join(kinect_save_path, "init_params.npz")):
        # load init shape & pose
        bot.print("{} : [Minimal] Load current init params".format(ymdhms_time()))
        kinect_solver.update_params(np.load(os.path.join(kinect_save_path, "init_params.npz")))
    else:
        # solve init shape
        bot.print("{} : [Minimal] Start solving init params...".format(ymdhms_time()))
        kienct_shape_params = []

        for i in range(window_len*2+1):
            result, info = kinect_loader[i]
            jnts_brg.init_input(np.mean([result["sub1_skeleton"], result["sub2_skeleton"]], axis=0), np.vstack([result["sub1_pcl"], result["sub2_pcl"]]))

            _jnts, _pcl = jnts_brg.map(KINECT_DATA)

            # o3d_plot([o3d_pcl(_jnts, [1,0,0]), o3d_pcl(result["optitrack"], [0,0,1]), o3d_pcl(_pcl, [0,1,0])])

            init_param = np.zeros(kinect_solver.model.n_pose + kinect_solver.model.n_coord + kinect_solver.model.n_glb + kinect_solver.model.n_shape)
            # translation
            init_param[:3] = -(_jnts.max(axis=0) + _jnts.min(axis=0))/2
            # rotation
            init_param[3] = np.pi/2
            kinect_solver.update_params(init_param)

            _, losses = kinect_solver.solve(_jnts, _pcl, "full", dbg_level=dbg_level, max_iter=60, losses_with_weights=losses_w)
            
            kienct_shape_params.append(kinect_solver.shape_params)
            del losses
    
        if device == "cpu":
            kinect_solver.shape_params = np.array(kienct_shape_params).mean(0)
        else:
            kinect_solver.shape_params = torch.vstack(kienct_shape_params).mean(0)
        kinect_solver.save_param(os.path.join(kinect_save_path, "init_params"))

    # init pose update losses
    losses_w = dict(
        kpts_losses=1,
        edge_losses=50,
        face_losses=50,
    )

    bot.print("{} : [Minimal] Losses: {}".format(ymdhms_time(), losses_w))

    opti_inputs = MinimalInput(opti_loader, jnts_brg.scale, OPTI_DATA)
    kinect_inputs = MinimalInput(kinect_loader, jnts_brg.scale, KINECT_DATA)

    kinect_inputs.jnts_source = KINECT_SUB1_SOURCE

    # Restor Opti
    opti_current_minimal = MinimalLoader(root_path)
    kinect_current_minimal = MinimalLoader(root_path, [dict(tag="minimal_k/param", ext=".npz"),])

    opti_start_idx = 0
    kinect_start_idx = 0

    for i in range(len(opti_loader)):
        result = opti_current_minimal.select_item_in_tag(i+int(kwargs.get("skip_head", 0)), "rid", "minimal/param")
        if len(result) == 0:
            break
        opti_start_idx = i

    for i in range(len(kinect_loader)):
        result = kinect_current_minimal.select_item_in_tag(i+int(kwargs.get("skip_head", 0)), "rid", "minimal/param")
        if len(result) == 0:
            break
        kinect_start_idx = i
    
    start_idx = min(opti_start_idx, kinect_start_idx)

    bot.print("{} : [Minimal] Configure start index {}".format(ymdhms_time(), start_idx))

    for i in range(start_idx, min(len(opti_loader), len(kinect_loader))):
        rid = i+int(kwargs.get("skip_head", 0))
        opti_init_pose = opti_solver.pose_params
        kinect_init_pose = kinect_solver.pose_params
        empty_pcl = np.array([[0,0,0]])
        opti_results = {}
        kinect_results = {}
        for j in range(max(0, i-window_len), min(len(opti_loader), len(kinect_loader), i+window_len+1)):
            opti_solver.update_params(opti_init_pose)
            pcl_source = opti_inputs[i]
            _, opti_losses = opti_solver.solve(opti_inputs[j]["jnts"], pcl_source["pcl"] if "pcl" in pcl_source.keys() else empty_pcl, "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)

            opti_results[j] = dict(
                pose = opti_solver.pose_params,
                loss = opti_losses[-1]
            )

            kinect_solver.update_params(kinect_init_pose)
            pcl_source = kinect_solver[i]
            _, kinect_losses = kinect_solver.solve(kinect_inputs[j]["jnts"], pcl_source["pcl"] if "pcl" in pcl_source.keys() else empty_pcl, "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)

            kinect_results[j] = dict(
                pose = kinect_solver.pose_params,
                loss = kinect_losses[-1],
                device = kinect_inputs.jnts_source
            )

            # if loss exceeds threshold, try switching to another skeleton source
            if kinect_losses[-1] > switch_skel_loss_threshold:
                kinect_inputs.jnts_source = not kinect_inputs.jnts_source
                _, new_losses = kinect_solver.solve(kinect_inputs[j]["jnts"], pcl_source["pcl"] if "pcl" in pcl_source.keys() else empty_pcl, "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)
                if new_losses[-1] < kinect_losses[-1]:
                    # switch to skeleton source of another device
                    kinect_results[j] = dict(
                        pose = kinect_solver.pose_params,
                        loss = new_losses[-1],
                        device = kinect_inputs.jnts_source
                    )
                else:
                    # use origin source
                    kinect_inputs.jnts_source = not kinect_inputs.jnts_source
                del new_losses

            del opti_losses, kinect_losses
        opti_result_key = min(opti_results, key=lambda key: opti_results[key]["loss"])
        kinect_result_key = min(kinect_results, key=lambda key: kinect_results[key]["loss"])

        opti_solver.update_params(opti_results[opti_result_key]["pose"])
        kinect_solver.update_params(kinect_results[kinect_result_key]["pose"])
        kinect_skel_source = "sub2" if kinect_results[kinect_result_key]["device"] == KINECT_SUB2_SOURCE else "sub1"
        
        if opti_inputs[j]["info"].get("nan", False):
            nan_flag = "nan"
        else:
            nan_flag = "fine"
        opti_filename = "id={}_skid={}_sub1id={}_rid={}_type={}".format(i, opti_inputs[j]["info"]["optitrack"]["id"], opti_inputs[i]["info"]["sub1_pcl"]["id"], rid, nan_flag)

        opti_solver.save_param(os.path.join(opti_save_path, "param", opti_filename))
        opti_solver.save_model(os.path.join(opti_save_path, "obj", opti_filename+".obj"))
        opti_inputs.remove(i-window_len)

        kinect_filename = "id={}_skid={}_{}id={}_rid={}_type={}".format(i, kinect_inputs[j]["info"]["{}_skeleton".format(kinect_skel_source)]["id"], kinect_skel_source, kinect_inputs[i]["info"]["{}_pcl".format(kinect_skel_source)]["id"], rid, nan_flag)
        kinect_solver.save_param(os.path.join(kinect_save_path, "param", kinect_filename))
        kinect_solver.save_model(os.path.join(kinect_save_path, "obj", kinect_filename+".obj"))
        kinect_inputs.remove(i-window_len)

        bot.print("{} : [Minimal] Opti {} Frame rid={} with loss {:.4}".format(ymdhms_time(), root_path[-21:-2], rid, opti_results[opti_result_key]["loss"]))
        bot.print("{} : [Minimal] Kinect {} Frame rid={} with loss {:.4}".format(ymdhms_time(), root_path[-21:-2], rid, opti_results[kinect_result_key]["loss"]))
        gc.collect()
    bot.add_task("{} : [Minimal] {} finished.".format(ymdhms_time(), root_path[-21:-2]), MSG_ERROR)
    sleep(15)


def stream_windowed_minimal(root_path: str, dbg_level: int=0, window_len: int=2, plot_type="open3d", device="cpu", data_type=OPTI_DATA, switch_skel_loss_threshold=1.0, **kwargs):
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.solver import Solver
    from minimal.models_torch import KinematicModel as KinematicModelTorch, KinematicPCAWrapper as KinematicPCAWrapperTorch
    from minimal.solver_torch import Solver as SolverTorch
    # from visualization.utils import o3d_plot, o3d_pcl
    
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
        edge_losses=50,
        face_losses=50,
        vposer_j_loss=1,
        vposer_v_loss=1,
        vposer_p_loss=1
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
                jnts_brg.init_input(result["optitrack"], np.vstack([result["sub1_pcl"], result["sub2_pcl"]]))
            elif data_type == KINECT_DATA:
                # input is mean of sub1 & sub2
                jnts_brg.init_input(np.mean([result["sub1_skeleton"], result["sub2_skeleton"]], axis=0), np.vstack([result["sub1_pcl"], result["sub2_pcl"]]))
            else:
                raise NotImplementedError

            _jnts, _pcl = jnts_brg.map(data_type)

            # o3d_plot([o3d_pcl(_jnts, [1,0,0]), o3d_pcl(result["optitrack"], [0,0,1]), o3d_pcl(_pcl, [0,1,0])])

            init_param = np.zeros(solver.model.n_pose + solver.model.n_coord + solver.model.n_glb + solver.model.n_shape)
            # translation
            init_param[:3] = -(_jnts.max(axis=0) + _jnts.min(axis=0))/2
            # rotation
            init_param[3] = np.pi/2
            solver.update_params(init_param)

            _, losses = solver.solve(_jnts, _pcl, "full", dbg_level=dbg_level, max_iter=60, losses_with_weights=losses_w)
            
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
        vposer_j_loss=1,
        vposer_v_loss=1,
        vposer_p_loss=1
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
        for j in range(max(0, i-window_len), min(len(loader), i+window_len+1)):
            solver.update_params(init_pose)
            pcl_source = inputs[i]
            _, losses = solver.solve(inputs[j]["jnts"], pcl_source["pcl"] if "pcl" in pcl_source.keys() else empty_pcl, "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)

            results[j] = dict(
                pose = solver.pose_params,
                loss = losses[-1],
                device = inputs.jnts_source
            )

            # if loss exceeds threshold, try switching to another skeleton source
            if data_type == KINECT_DATA and losses[-1] > switch_skel_loss_threshold:
                # switch to skeleton source of another device
                inputs.jnts_source = not inputs.jnts_source
                _, new_losses = solver.solve(inputs[j]["jnts"], pcl_source["pcl"] if "pcl" in pcl_source.keys() else empty_pcl, "pose", max_iter=40, kpts_threshold=0.02, loss_threshold=0.0005, mse_threshold=0.0001, dbg_level=dbg_level, losses_with_weights=losses_w)
                if new_losses[-1] < losses[-1]:
                    results[j] = dict(
                        pose = solver.pose_params,
                        loss = new_losses[-1],
                        device = inputs.jnts_source
                    )
                else:
                    # use origin source
                    inputs.jnts_source = 1 - inputs.jnts_source
                del new_losses

            del losses
        result_key = min(results, key=lambda key: results[key]["loss"])
        solver.update_params(results[result_key]["pose"])

        if inputs[j]["info"].get("nan", False):
            nan_flag = "nan"
        else:
            nan_flag = "fine"
        
        if data_type == OPTI_DATA:
            filename = "id={}_skid={}_sub1id={}_rid={}_type={}".format(i, inputs[j]["info"]["optitrack"]["id"], inputs[i]["info"]["sub1_pcl"]["id"], rid, nan_flag)
        elif data_type == KINECT_DATA:
            kinect_skel_source = "sub2" if results[result_key]["device"] == KINECT_SUB2_SOURCE else "sub1"
            filename = "id={}_skid={}_{}id={}_rid={}_type={}".format(i, inputs[j]["info"]["{}_skeleton".format(kinect_skel_source)]["id"], kinect_skel_source, inputs[i]["info"]["{}_pcl".format(kinect_skel_source)]["id"], rid, nan_flag)

        solver.save_param(os.path.join(save_path, "param", filename))
        solver.save_model(os.path.join(save_path, "obj", filename+".obj"))
        # inputs.save_revert_transform(j, os.path.join(save_path, "trans", filename))
        inputs.remove(i-window_len)

        bot.print("{} : [Minimal] {} Frame rid={} with loss {:.4}".format(ymdhms_time(), root_path[-21:-2], rid, results[result_key]["loss"]))
        gc.collect()
    bot.add_task("{} : [Minimal] {} finished.".format(ymdhms_time(), root_path[-21:-2]), MSG_ERROR)
    sleep(15)


def check_input(root_path, **kwargs):
    from visualization.mesh_plot import MinimalInputStreamPlot

    MinimalInputStreamPlot(root_path, skip_head=int(kwargs.get("skip_head", 0)), skip_tail=int(kwargs.get("skip_tail", 0))).show()


def check_result(root_path, **kwargs):
    from visualization.mesh_plot import MinimalResultStreamPlot

    MinimalResultStreamPlot(root_path, skip_head=kwargs.get("skip_head", 0), skip_tail=kwargs.get("skip_tail", 0)).show()


def check_save_parallel(root_path, **kwargs):
    from visualization.mesh_plot import MinimalResultParallelPlot

    MinimalResultParallelPlot(root_path, pause=kwargs.pop("pause", True), save_path=kwargs.pop("save_path", "./"), skip_head=kwargs.get("skip_head", 0), skip_tail=kwargs.get("skip_tail", 0)).show()


def check_save_result(root_path, **kwargs):
    from visualization.mesh_plot import MeshPclStreamPlot

    MeshPclStreamPlot(root_path, save_path=kwargs.pop("save_path", "./"), skip_head=kwargs.get("skip_head", 0), skip_tail=kwargs.get("skip_tail", 0)).show()


def run():
    task_dict = dict(
        null=exit,
        kinect_single_minimal=kinect_single_frame,
        optitrack_single_minimal=optitrack_single_frame,
        parallel_single_minimal=parallel_single_frame,
        stream_windowed=stream_windowed_minimal,
        parallel_stream_windowed=stream_kinect_optitrack_windowed,
        check_input=check_input,
        check_result=check_result,
        check_save_result=check_save_result,
        check_save_parallel=check_save_parallel,
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
    
    device = args_dict.get("device", "gpu:-1")
    if ':' not in device:
        device += ":-1"

    args_dict["device"], device_id = device.split(":")

    if int(device_id) not in range(get_gpu_count()):
        gpu_id = get_freer_gpu()
        print("Auto select GPU {}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    else:
        print("Custom GPU {}".format(device_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    try:
        task_dict[args.task](**args_dict)
    except Exception as e:
        if int(args_dict.get("dingtalk", 1)):
            args_dict["msg_bot"].enable()
            args_dict["msg_bot"].add_task("{} : task={}, path={}\ndetails: {}".format(ymdhms_time, args.task, args.root_path, e), MSG_ERROR)
            sleep(15)
        raise e

if __name__ == "__main__":
    run()
