from visualization.utils import O3DStreamPlot, o3d_coord, o3d_pcl, o3d_skeleton, pcl_filter
import numpy as np
import open3d as o3d
from itertools import compress
from dataloader.result_loader import KinectResultLoader, ArbeResultLoader, OptitrackResultLoader


class SkelArbeManager():
    def __init__(self, result_path, *devices) -> None:
        if not devices:
            devices = ("master","sub1","sub2")
        self.devices = devices
        self.k_loader_dict = {}
        for device in devices:
            param = [dict(tag="kinect/{}/skeleton".format(device), ext=".npy")]
            self.k_loader_dict.update({device:KinectResultLoader(result_path, param)})
        self.a_loader = ArbeResultLoader(result_path)

    def generator(self, device="master"):
        if device not in self.devices:
            print("No {}".format(device))
            exit(1)
        param = "kinect/{}/skeleton".format(device)
        for i in range(len(self.a_loader)):
            a_row = self.a_loader[i]
            k_row = self.k_loader_dict[device].select_item(a_row["arbe"]["tm"], "st", False)
            # k_row = self.k_loader_dict[device].select_by_skid(i)
            # a_row = self.a_loader.select_item(k_row[param]["st"], "tm", False)
            yield k_row[param], a_row["arbe"]


class KinectArbeStreamPlot(O3DStreamPlot):
    def __init__(self, input_path: str, devices: list = ['master'], *args, **kwargs) -> None:
        super().__init__(width=800, *args, **kwargs)
        self.input_path = input_path
        self.devices = devices

    def init_updater(self):
        self.plot_funcs = dict(
            kinect_skeleton=o3d_skeleton,
            kinect_pcl=o3d_pcl,
            arbe_pcl=o3d_pcl,
        )

    def generator(self, device: str = None):
        if device is None:
            device = self.devices[0]
        input_manager = SkelArbeManager(self.input_path, *self.devices)
        for kinect_row, arbe_row in input_manager.generator(device):
            # load numpy from file
            kinect_arr = np.load(kinect_row["filepath"])
            arbe_arr = np.load(arbe_row["filepath"])

            person_count = kinect_arr.shape[0]
            kinect_skeleton = kinect_arr[:,:,:3].reshape(-1,3)/1000

            # transform
            # TODO: update transaction accorging to skeleton
            rotation = np.array([[1,0,0],
                                [0,0,-1],
                                [0,1,0]])
            translation = np.array([0, 0, 0.2]).T
            skeleton_pcl = kinect_skeleton @ rotation + translation

            # filter pcl with naive bounding box
            arbe_pcl = pcl_filter(skeleton_pcl, arbe_arr[:,:3], 0.5)

            # init lines
            lines = np.asarray([[0,1],[1,2],[2,3],[2,4],[4,5],[5,6],[6,7],[7,8],
                                [8,9],[7,10],[2,11],[11,12],[12,13],[13,14],[14,15],
                                [15,16],[14,17],[0,18],[18,19],[19,20],[20,21],[0,22],
                                [22,23],[23,24],[24,25],[3,26],[26,27],[26,28],[26,29],
                                [26,30],[26,31]])
            if person_count > 1:
                for p in range(1, person_count):
                    lines = np.vstack((lines, lines + p*32))

            lines_colors = np.array([[0, 0, 1] for j in range(len(lines))])
            
            yield dict(
                kinect_skeleton=dict(
                    skeleton=skeleton_pcl,
                    lines=lines,
                    lines_colors=lines_colors
                ),
                kinect_pcl=dict(
                    pcl=skeleton_pcl,
                    color=[0,0,1]
                ),
                arbe_pcl=dict(
                    pcl=arbe_pcl,
                    color=[0,1,0]
                ),
            )


class OptiArbeManager():
    def __init__(self, result_path) -> None:
        self.opti_loader = OptitrackResultLoader(result_path)
        self.arbe_loader = ArbeResultLoader(result_path)

    def generator(self):
        for i in range(len(self.arbe_loader)):
            arbe_row = self.arbe_loader[i]
            opti_row = self.opti_loader["optitrack"].select_item(arbe_row["arbe"]["tm"], "tm", False)
            yield opti_row["optitrack"], arbe_row["arbe"]


class OptitrackArbeStreamPlot(O3DStreamPlot):
    def __init__(self, input_path: str, *args, **kwargs) -> None:
        super().__init__(width=800, *args, **kwargs)
        self.input_path = input_path

    def init_updater(self):
        self.plot_funcs = dict(
            opti_skeleton=o3d_skeleton,
            arbe_pcl=o3d_pcl,
        )

    def generator(self):
        input_manager = OptiArbeManager(self.input_path)
        for opti_row, arbe_row in input_manager.generator():
            # load numpy from file
            opti_arr_dict = np.load(opti_row["filepath"])
            arbe_arr = np.load(arbe_row["filepath"])

            person_count = opti_arr_dict["markers"].shape[0]
            markers_pcl = opti_arr_dict["markers"][:,:,:3].reshape(-1,3)
            bones_pcl = opti_arr_dict["bones"][:,:,:3].reshape(-1,3)

            # transform
            # TODO: update transaction accorging to skeleton
            rotation = np.array([[1,0,0],
                                [0,0,-1],
                                [0,1,0]])
            translation = np.array([0, 0, 0.2]).T
            opti_markers = markers_pcl @ rotation + translation
            opti_bones = bones_pcl @ rotation + translation

            # filter pcl with naive bounding box
            arbe_pcl = pcl_filter(opti_markers, arbe_arr[:,:3], 0.5)

            # init lines
            # TODO: update the lines
            lines = np.asarray([[0,1],[1,2],[2,3],[2,4],[4,5],[5,6],[6,7],[7,8],
                                [8,9],[7,10],[2,11],[11,12],[12,13],[13,14],[14,15],
                                [15,16],[14,17],[0,18],[18,19],[19,20],[20,21],[0,22],
                                [22,23],[23,24],[24,25],[3,26],[26,27],[26,28],[26,29],
                                [26,30],[26,31]])
            if person_count > 1:
                for p in range(1, person_count):
                    lines = np.vstack((lines, lines + p*37))

            lines_colors = np.array([[0, 0, 1] for j in range(len(lines))])
            
            yield dict(
                opti_markers=dict(
                    markers=opti_markers,
                    lines=lines,
                    lines_colors=lines_colors
                ),
                opti_bones=dict(
                    bones=opti_bones,
                    colors=[1,0,0]
                ),
                arbe_pcl=dict(
                    pcl=arbe_pcl,
                    color=[0,1,0]
                ),
            )


def plot_minimal_input(jnts_input, pcl_input, jnts_smpl, mesh_smpl):
    from visualization.utils import o3d_plot, o3d_pcl, o3d_mesh
    o3d_plot([o3d_pcl(jnts_input, [0,0,1]), o3d_pcl(pcl_input, [1,0,0]), o3d_pcl(jnts_smpl, [0,1,0]), o3d_mesh(mesh_smpl, [1,1,0])], 'Minimal Input')
