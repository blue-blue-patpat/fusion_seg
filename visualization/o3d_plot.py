from cv2 import transform
import kinect
from torch._C import R
from visualization.utils import O3DStreamPlot, o3d_coord, o3d_pcl, o3d_skeleton, pcl_filter
import numpy as np
import open3d as o3d
from itertools import compress
from dataloader.result_loader import KinectResultLoader, ArbeResultLoader, OptitrackResultLoader
from kinect.config import EXTRINSIC_MAS_ARBE, KINECT_SKELETON_LINES


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
            kinect_arbe_matix = EXTRINSIC_MAS_ARBE
            skeleton_pcl = kinect_skeleton @ kinect_arbe_matix["R"].T + kinect_arbe_matix["t"]

            # filter pcl with naive bounding box
            arbe_pcl = pcl_filter(skeleton_pcl, arbe_arr[:,:3], 0.5)

            # init lines
            lines = KINECT_SKELETON_LINES
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


class KinectRealtimeStreamPlot(O3DStreamPlot):
    
    def __init__(self, input_path: str, devices: list = ['master'], *args, **kwargs) -> None:
        from kinect.config import MAS, SUB1, SUB2, INTRINSIC
        super().__init__(width=800, *args, **kwargs)
        self.input_path = input_path
        self.devices = devices
        self.devices_type = dict(master="mas", sub1="sub", sub2="sub")
        self.devices_id = dict(master=MAS, sub1=SUB1, sub2=SUB2)

    def init_updater(self):
        self.plot_funcs = {}
        for device_name in self.devices:
            self.plot_funcs[device_name] = o3d_pcl

    def generator(self, root_path: str = None):
        from multiprocessing.dummy import Pool
        from pyk4a import PyK4A
        import cv2
        from dataloader.kinect_loader import _get_config, _get_device_ids
        from calib.utils import kinect_transform_mat

        if root_path is None:
            root_path = "./__test__/default"

        started_devices = {}

        for device_name in self.devices:
            started_devices[device_name] = PyK4A(config=_get_config(self.device_type[device_name]), device_id=_get_device_ids()[self.devices_id[device_name]])
            started_devices[device_name].start()

        def process(device, R, t):
            # skip first 60 frames
            for i in range(60):
                device.get_capture()
            
            capture = device.get_capture()
            # device.close()
            color_frame, pcl_frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB), capture.transformed_depth_point_cloud
            pcl = o3d_pcl(pcl_frame.reshape(-1,3)/1000 @ R.T + t)
            pcl.colors = o3d.utility.Vector3dVector(color_frame.reshape(-1, 3)/255)
            return pcl

        transform_mats = kinect_transform_mat()["kinect_{}".format(device_name)]

        R = transform_mats["R"]
        t = transform_mats["t"]

        pool = Pool()
        
        while True:
            results = {}
            for device_name in self.devices:
                results[device_name] = dict(pcl=pool.apply_async(process, (started_devices[device_name], R, t)).get())
            yield results


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
        self.calib_mtx = np.load(self.input_path + "/calib/optitrack/transform.npz")

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
            opti_markers = markers_pcl @ self.calib_mtx["R"] + self.calib_mtx["t"]
            opti_bones = bones_pcl @ self.calib_mtx["R"] + self.calib_mtx["t"]

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
