import time, os
from typing import Callable, Dict, Generator, Iterable, Union, overload
import numba as nb
import numpy as np
import open3d as o3d
from pytorch3d.structures import Meshes
import torch


def o3d_plot(o3d_items: list, title="", show_coord=True, **kwargs):
    if show_coord:
        _items = o3d_items + [o3d_coord(**kwargs)]
    else:
        _items = o3d_items
    view = o3d.visualization.VisualizerWithKeyCallback()
    view.create_window()
    # render = view.get_render_option()
    # render.point_size = 0.5
    for item in _items:
        view.add_geometry(item)
    view.run()
    # o3d.visualization.draw_geometries(_items, title)


def o3d_coord(size=0.1, origin=[0, 0, 0]):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


def o3d_pcl(pcl: np.ndarray = None, color: list = None, colors: list = None, last_update = None):
    _pcl = last_update
    if _pcl is None:
        _pcl = o3d.geometry.PointCloud()

    if pcl is not None and pcl.size != 0:
        if pcl.shape[0] > 1000000:
            # auto downsample
            pcl = pcl[np.random.choice(np.arange(pcl.shape[0]), size=1000000, replace=False)]
        _pcl.points = o3d.utility.Vector3dVector(pcl)
        if color is not None:
            _pcl.paint_uniform_color(color)
        if colors is not None:
            _pcl.colors = o3d.utility.Vector3dVector(colors)
    return _pcl


def o3d_box(upper_bounds: np.ndarray = None, lower_bounds: np.ndarray = None, color: list = [1,0,0], last_update = None):
    _box = last_update
    if _box is None:
        _box = o3d.geometry.LineSet()
        _box.lines = o3d.utility.Vector2iVector(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7]
            ]
        )
    
    if upper_bounds is not None and lower_bounds is not None:
        x_u, y_u, z_u = upper_bounds
        x_l, y_l, z_l = lower_bounds
        _box.points = o3d.utility.Vector3dVector(
            [
                [x_u, y_u, z_u],
                [x_l, y_u, z_u],
                [x_l, y_l, z_u],
                [x_u, y_l, z_u],
                [x_u, y_u, z_l],
                [x_l, y_u, z_l],
                [x_l, y_l, z_l],
                [x_u, y_l, z_l],
            ]
        )
    if color is None:
        colors = np.repeat([color], 8, axis=0)
        _box.colors = o3d.utility.Vector3dVector(colors)
    return _box


def o3d_skeleton(skeleton: np.ndarray = None, lines: np.ndarray = None,
                 color: list = [1,0,0], colors: list = None,
                 last_update = None):
    _lines = last_update
    if _lines is None:
        _lines = o3d.geometry.LineSet()

    if skeleton is not None:
        _lines.points = o3d.utility.Vector3dVector(skeleton)
    if lines is not None:
        _lines.lines = o3d.utility.Vector2iVector(lines)
        if colors is None:
            colors = np.repeat([color], lines.shape[0], axis=0)
        _lines.colors = o3d.utility.Vector3dVector(colors)
    return _lines


@overload
def o3d_mesh(mesh: Meshes, color: list, last_update: None): ...


@overload
def o3d_mesh(mesh: Iterable, color: list, last_update: None): ...


def o3d_mesh(mesh: Union[Meshes, Iterable] = None, color: list = None,
             last_update = None):
    _mesh = last_update
    if _mesh is None:
        _mesh = o3d.geometry.TriangleMesh()

    if mesh is not None:
        if isinstance(mesh, Meshes):
            _mesh.vertices = o3d.utility.Vector3dVector(mesh.verts_packed().cpu())
            _mesh.triangles = o3d.utility.Vector3iVector(mesh.faces_packed().cpu())
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            _mesh.vertices = mesh.vertices
            _mesh.triangles = mesh.triangles
        else:
            _mesh.vertices = o3d.utility.Vector3dVector(mesh[0])
            if mesh[1] is not None:
                _mesh.triangles = o3d.utility.Vector3iVector(mesh[1])
        if color is not None:
            _mesh.paint_uniform_color(color)
        _mesh.compute_vertex_normals()
    return _mesh


def o3d_smpl_mesh(params: Union[np.ndarray, np.lib.npyio.NpzFile, dict] = None, color: list = None,
             model = None, device = "cpu", last_update = None):
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.models_torch import KinematicModel as KinematicModelTorch, KinematicPCAWrapper as KinematicPCAWrapperTorch
    import minimal.config as config
    import minimal.armatures as armatures

    _mesh = last_update
    if _mesh is None:
        _mesh = o3d.geometry.TriangleMesh()

    if params is None:
        return o3d_mesh(None, color, _mesh)
    
    if model is None:
        if device == "cpu":
            model = KinematicPCAWrapper(KinematicModel().init_from_file(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature, compute_mesh=False))
        else:
            model = KinematicPCAWrapperTorch(KinematicModelTorch(torch.device(device)).init_from_file(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature, compute_mesh=False))

    assert isinstance(model, (KinematicPCAWrapper, KinematicPCAWrapperTorch)), "Undefined model"

    # Unzip dict params
    if isinstance(params, np.lib.npyio.NpzFile) or (isinstance(params, dict) and isinstance(params.values()[0], np.ndarray)):
        # Numpy dict input
        pose_params = params["pose"]
        shape_params = params["shape"]
        params = np.hstack([pose_params, shape_params])

    elif isinstance(params, dict) and isinstance(params.values()[0], torch.Tensor):
        # Torch Tensor dict input
        pose_params = params["pose"]
        shape_params = params["shape"]
        params = torch.cat([pose_params, shape_params]).to(torch.float64)

    # Convert unmatched data types
    if isinstance(model, KinematicPCAWrapper) and isinstance(params, torch.Tensor):
        params = params.cpu().numpy()
    elif isinstance(model, KinematicModelTorch) and isinstance(params, np.ndarray):
        params = torch.from_numpy(params).to(device=model.device, dtype=torch.float64)

    model.run(params)

    if isinstance(model, KinematicPCAWrapper):
        mesh = [model.core.verts, model.core.faces]
    else:
        mesh = [model.core.verts.cpu().numpy(), model.core.faces.cpu().numpy()]

    return o3d_mesh(mesh, color, last_update=_mesh)


def pcl_filter(pcl_box, pcl_target, bound=0.2, offset=0):
    """
    Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
    """
    upper_bound = pcl_box[:,:3].max(axis=0) + bound
    lower_bound = pcl_box[:,:3].min(axis=0) - bound
    lower_bound[2] += offset

    mask_x = (pcl_target[:, 0] >= lower_bound[0]) & (pcl_target[:, 0] <= upper_bound[0])
    mask_y = (pcl_target[:, 1] >= lower_bound[1]) & (pcl_target[:, 1] <= upper_bound[1])
    mask_z = (pcl_target[:, 2] >= lower_bound[2]) & (pcl_target[:, 2] <= upper_bound[2])
    index = mask_x & mask_y & mask_z
    return pcl_target[index]


@nb.jit
def filter2_np_nb(arr: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray):
    n = 0
    flag = True
    for i in nb.prange(0, arr.shape[0]):
    # for i in range(arr.shape[0]):
        flag = True
        for j in range(lower_bound.size):
            if not (arr[i][j] > lower_bound[j] and arr[i][j] < upper_bound[j]):
                flag = False
                break
        if flag:
            n += 1
    result = np.empty((n, arr.shape[1]), dtype=arr.dtype)
    _n = 0
    for i in nb.prange(0, arr.shape[0]):
        flag = True
        for j in range(lower_bound.size):
            if not (arr[i][j] > lower_bound[j] and arr[i][j] < upper_bound[j]):
                flag = False
                break
        if flag:
            result[_n] = arr[i]
            _n += 1
        if _n == n:
            break
    return result


def pcl_filter_nb(pcl_box, pcl_target, bound=0.5):
    """
    Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
    """

    upper_bound = pcl_box[:,:3].max(axis=0) + bound
    lower_bound = pcl_box[:,:3].min(axis=0) - bound

    return filter2_np_nb(pcl_target, lower_bound, upper_bound)

def pcl_filter_nb_noground(pcl_box, pcl_target, bound=0.5):
    """
    Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
    """

    upper_bound = pcl_box[:,:3].max(axis=0) + bound
    lower_bound = pcl_box[:,:3].min(axis=0) - bound
    lower_bound[2] = lower_bound[2] + 0.21

    return filter2_np_nb(pcl_target, lower_bound, upper_bound)


from pyk4a import CalibrationType, PyK4APlayback
import cv2
from torchvision import models

resnet18 = models.resnet18(pretrained=True)
modules = list(resnet18.children())[:-2]
modules.append(torch.nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True))
model = torch.nn.Sequential(*modules)

def get_feature_map(img):
    with torch.no_grad():
        img = cv2.resize(img, (224, 224))
        input = img.transpose(2, 0, 1)
        input = torch.tensor(input, dtype=torch.float)[None]
        
        output = model(input)
        output = output[0].numpy().transpose(1, 2, 0)
        channel_r = np.sum(output[:,:,:170], -1)
        channel_g = np.sum(output[:,:,170:340], -1)
        channel_b = np.sum(output[:,:,340:], -1)
        output = np.stack((channel_r, channel_g, channel_b), -1)

    return output

def get_image_feature(img):
    with torch.no_grad():
        img = cv2.resize(img, (224, 224))
        input = img.transpose(2, 0, 1)
        input = torch.tensor(input, dtype=torch.float)[None]
        
        output = resnet18(input)

    return output.numpy()

def image_crop(skeleton, image, playback=None, visual=False):
    skel_max = skeleton.max(axis=0) + 0.2
    skel_min = skeleton.min(axis=0) - 0.2
    box_3d = [
    [skel_min[0], skel_min[1], skel_min[2]],
    [skel_min[0], skel_min[1], skel_max[2]],
    [skel_min[0], skel_max[1], skel_max[2]],
    [skel_min[0], skel_max[1], skel_min[2]],
    [skel_max[0], skel_max[1], skel_max[2]],
    [skel_max[0], skel_min[1], skel_max[2]],
    [skel_max[0], skel_max[1], skel_min[2]],
    [skel_max[0], skel_min[1], skel_min[2]],
    ]
    box_2d = []

    for p in box_3d:
        if playback is not None:
            box_2d.append(playback.calibration.convert_3d_to_2d(p, CalibrationType.COLOR))
        else:
            box_2d.append((INTRINSIC[MAS] @ p/p[2])[:2])
    box_2d = np.floor(box_2d).astype(int)
    box_2d[:,[0,1]] = box_2d[:,[1,0]]
    box_min = box_2d.min(0)
    box_max = box_2d.max(0)
    crop_img = image[box_min[0]:box_max[0], box_min[1]:box_max[1]]

    if visual:
        cv2.namedWindow('img', 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.rectangle(image, box_min[::-1], box_max[::-1], (0, 0, 255), 2)
        cv2.imshow('img', image)
        cv2.waitKey(0)
    
    return crop_img, box_min, box_max

from kinect.config import *
def pcl_project(pcl, playback=None):
    pcl_2d = []
    for p in pcl:
        if playback is not None:
            point_2d = playback.calibration.convert_3d_to_2d(p, CalibrationType.COLOR)
        else:
            point_2d = (INTRINSIC[MAS] @ p/p[2])[:2]
        pcl_2d.append(point_2d)
    pcl_2d = np.asarray(pcl_2d)
    pcl_2d[:,[0,1]] = pcl_2d[:,[1,0]]
    
    return pcl_2d

def get_rgb_feature(pcl, image, mkv_fname=None, visual=False):
    if mkv_fname is not None:
        playback = PyK4APlayback(mkv_fname)
        playback.open()
    else:
        playback = None
    pcl_2d = pcl_project(pcl, playback)
    pcl_2d = np.floor(pcl_2d).astype(int)

    pcl_color = image[pcl_2d[:,0], pcl_2d[:,1]]
    pcl_with_feature = np.hstack((pcl, pcl_color/255))
    
    if visual:
        image[pcl_2d[:,0],pcl_2d[:,1]] = [0, 255, 0]
        cv2.namedWindow('img', 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.imshow('img', image)
        cv2.waitKey(0)

    return pcl_with_feature

def get_pcl_feature(pcl, image, skeleton, feature_type, mkv_fname=None, visual=False):
    if mkv_fname is not None:
        playback = PyK4APlayback(mkv_fname)
        playback.open()
    else:
        playback = None
    pcl_2d = pcl_project(pcl, playback)
    pcl_2d = np.floor(pcl_2d).astype(int)

    if feature_type == 'rgb':
        pcl_color = image[pcl_2d[:,0], pcl_2d[:,1]]
        pcl_with_feature = np.hstack((pcl, pcl_color/255))
    
    elif feature_type == 'feature_map':
        crop_img, box_min, _ = image_crop(skeleton, image, playback, visual)
        feature_map = get_feature_map(crop_img/255)
        feature_map = cv2.resize(feature_map, crop_img.shape[1::-1])
        feature = feature_map[pcl_2d[:,0]-box_min[0], pcl_2d[:,1]-box_min[1]]
        pcl_with_feature = np.hstack((pcl, feature))
    
    else:
        crop_img, box_min, _ = image_crop(skeleton, image, playback, visual)
        feature = get_image_feature(crop_img/255)
        pcl_with_feature = np.hstack((pcl, np.repeat(feature, pcl.shape[0], axis=0)))

    if visual:
        image[pcl_2d[:,0],pcl_2d[:,1]] = [0, 255, 0]
        cv2.namedWindow('img', 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.imshow('img', image)
        cv2.waitKey(0)

    return pcl_with_feature


class O3DItemUpdater():
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.update_item = func()

    def update(self, params: dict):
        self.func(last_update = self.update_item, **params)

    def get_item(self):
        return self.update_item


class O3DStreamPlot():
    pause = False
    speed_rate = 1
    def __init__(self, width=1600, height=1200, with_coord=True) -> None:
        self.view = o3d.visualization.VisualizerWithKeyCallback()
        self.view.create_window(width=width, height=height)
        self.ctr = self.view.get_view_control()
        self.render = self.view.get_render_option()
        try:
            self.render.point_size = 3.0
        except:
            print('No render setting')

        self.with_coord = with_coord
        self.first_render = True

        self.plot_funcs = dict()
        self.updater_dict = dict()
        self.init_updater()
        self.init_plot()
        self.init_key_cbk()

    def init_updater(self):
        self.plot_funcs = dict(exampel_pcl=o3d_pcl, example_skeleton=o3d_skeleton, example_mesh=o3d_mesh)
        raise RuntimeError("'O3DStreamPlot.init_updater' method should be overriden")

    def init_plot(self):
        for updater_key, func in self.plot_funcs.items():
            updater = O3DItemUpdater(func)
            self.view.add_geometry(updater.get_item())
            if self.with_coord:
                self.view.add_geometry(o3d_coord())
            self.updater_dict[updater_key] = updater

    def init_key_cbk(self):
        key_map = dict(
            w=87, a=65, s=83, d=68, h=72, l=76, space=32, one=49, two=50, four=52
        )
        key_cbk = dict(
            w=lambda v : v.get_view_control().rotate(0, 40),
            a=lambda v : v.get_view_control().rotate(40, 0),
            s=lambda v : v.get_view_control().rotate(0, -40),
            d=lambda v : v.get_view_control().rotate(-40, 0),
            h=lambda v : v.get_view_control().scale(-2),
            l=lambda v : v.get_view_control().scale(2),
            space=lambda v : exec("O3DStreamPlot.pause = not O3DStreamPlot.pause"),
            one=lambda v : exec("O3DStreamPlot.speed_rate = 1"),
            two=lambda v : exec("O3DStreamPlot.speed_rate = 2"),
            four=lambda v : exec("O3DStreamPlot.speed_rate = 4"),
        )

        for key, value in key_map.items():
            self.view.register_key_callback(value, key_cbk[key])

    def init_show(self):
        self.view.reset_view_point(True)
        self.first_render = False

    def update_plot(self):
        self.view.update_geometry(None)
        if self.first_render:
            self.init_show()
        self.view.poll_events()
        self.view.update_renderer()

    def show(self, gen: Generator = None, fps: float=30, save_path: str = ''):
        print("[O3DStreamPlot] rotate: W(left)/A(up)/S(down)/D(right); resize: L(-)/H(+); pause/resume: space; speed: 1(1x)/2(2x)/4(4x)")
        
        if gen is None:
            gen = self.generator()
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        tick = time.time()
        frame_idx = 0
        while True:
            duration = 1/(fps*self.speed_rate)
            while time.time() - tick < duration:
                continue

            # print("[O3DStreamPlot] {} FPS".format(1/(time.time() - tick)))

            tick = time.time()

            try:
                if not self.pause:
                    update_dict = next(gen)
            except StopIteration as e:
                break

            for updater_key, update_params in update_dict.items():
                if updater_key not in self.updater_dict.keys():
                    continue
                self.updater_dict[updater_key].update(update_params)
            self.update_plot()
            if save_path:
                self.view.capture_screen_image(os.path.join(save_path, '{}.png'.format(frame_idx)),True)
            frame_idx += 1

        # self.close_view()

    def show_manual(self, update_dict):
        for updater_key, update_params in update_dict.items():
            if updater_key not in self.updater_dict.keys():
                continue
            self.updater_dict[updater_key].update(update_params)
        self.update_plot()

    def close_view(self):
        self.view.close()
        self.view.destroy_window()

    def generator(self):
        raise RuntimeError("'O3DStreamPlot.generator' method should be overriden")
