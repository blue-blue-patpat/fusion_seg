import time
from typing import Callable, Dict, Generator, Iterable, Union, overload
import numpy as np
import open3d as o3d
from pytorch3d.structures import Meshes


def o3d_plot(o3d_items: list, title="", show_coord=True, **kwargs):
    if show_coord:
        _items = o3d_items + [o3d_coord(**kwargs)]
    else:
        _items = o3d_items
    o3d.visualization.draw_geometries(_items, title)


def o3d_coord(size=0.3, origin=[0, 0, 0]):
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
            colors = np.repeat([color], skeleton.shape[0], axis=0)
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
        else:
            _mesh.vertices = o3d.utility.Vector3dVector(mesh[0])
            _mesh.triangles = o3d.utility.Vector3iVector(mesh[1])
        if color is not None:
            _mesh.paint_uniform_color(color)
        _mesh.compute_vertex_normals()
    return _mesh


def pcl_filter(pcl_a, pcl_b, bound=0.5):
    """
    Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
    """
    from itertools import compress

    upper_bound = pcl_a[:,:3].max(axis=0) + bound
    lower_bound = pcl_a[:,:3].min(axis=0) - bound
    pcl_in_bound = (pcl_b[:,:3] < upper_bound) & (pcl_b[:,:3] > lower_bound)

    filter_list = []
    for row in pcl_in_bound:
        filter_list.append(False if False in row else True)
    return np.array(list(compress(pcl_b, filter_list)))


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
    def __init__(self, width=1600, height=1200, with_coord=True) -> None:
        self.view = o3d.visualization.VisualizerWithKeyCallback()
        self.view.create_window(width=width, height=height)
        self.ctr = self.view.get_view_control()

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
            w=87, a=65, s=83, d=68, h=72, l=76, space=32
        )
        key_cbk = dict(
            w=lambda v : v.get_view_control().rotate(0, 40),
            a=lambda v : v.get_view_control().rotate(40, 0),
            s=lambda v : v.get_view_control().rotate(0, -40),
            d=lambda v : v.get_view_control().rotate(-40, 0),
            h=lambda v : v.get_view_control().scale(-2),
            l=lambda v : v.get_view_control().scale(2),
            space=lambda v : exec("O3DStreamPlot.pause = not O3DStreamPlot.pause"),
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

    def show(self, gen: Generator = None, fps: float=30):
        if gen is None:
            gen = self.generator()
        tick = time.time()
        duration = 1/fps
        while True:
            while time.time() - tick < duration:
                time.sleep(0.002)
            
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
            tick = time.time()

        self.close_view()

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
