from visualization.utils import O3DStreamPlot, o3d_coord, o3d_pcl, o3d_plot, o3d_skeleton, pcl_filter
import numpy as np
import open3d as o3d
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
        for i in range(90, len(self.a_loader)):
            a_row = self.a_loader[i]
            k_row = self.k_loader_dict[device].select_item(a_row["arbe"]["st"], "st", False)
            # k_row = self.k_loader_dict[device].select_by_skid(i)
            # a_row = self.a_loader.select_item(k_row[param]["st"], "tm", False)
            yield k_row[param], a_row["arbe"]


class KinectArbeStreamPlot(O3DStreamPlot):
    def __init__(self, input_path: str, devices: list = ['master'], angle_of_view=[0,-1,0,1], *args, **kwargs) -> None:
        super().__init__(width=800, *args, **kwargs)
        self.input_path = input_path
        self.devices = devices
        self.angle_of_view = angle_of_view

    def init_updater(self):
        self.plot_funcs = dict(
            kinect_skeleton=o3d_skeleton,
            kinect_pcl=o3d_pcl,
            arbe_pcls=o3d_pcl,
        )
    
    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([0, 0, 1]))
        self.ctr.set_front(np.array(self.angle_of_view[:3]))
        self.ctr.set_zoom(self.angle_of_view[3])

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

            colors = np.array([[0, 0, 1] for j in range(len(lines))])
            
            yield dict(
                kinect_skeleton=dict(
                    skeleton=skeleton_pcl,
                    lines=lines,
                    colors=colors
                ),
                kinect_pcl=dict(
                    pcl=skeleton_pcl,
                    color=[0,0,1]
                ),
                arbe_pcls=dict(
                    pcl=arbe_pcl,
                    color=[0,1,0]
                ),
            )


class KinectRealtimeStreamPlot(O3DStreamPlot):
    def __init__(self, input_path: str, devices: list = ['master'], *args, **kwargs) -> None:
        from kinect.config import MAS, SUB1, SUB2, INTRINSIC
        self.input_path = input_path
        self.devices = devices
        self.devices_type = dict(master="mas", sub1="sub", sub2="sub")
        self.devices_id = dict(master=MAS, sub1=SUB1, sub2=SUB2)
        super().__init__(width=2000, *args, **kwargs)

    def init_updater(self):
        self.plot_funcs = {}
        for device_name in self.devices:
            self.plot_funcs[device_name] = o3d_pcl

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([[0],[0],[1]]))
        self.ctr.set_front(np.array([[0],[1],[0]]))
        self.ctr.set_zoom(0.3)

    def close_view(self):
        for device_name in self.devices:
            self.started_devices[device_name].close()
        super().close_view()

    def generator(self, root_path: str = None):
        from multiprocessing.dummy import Pool
        from pyk4a import PyK4A
        import cv2
        from dataloader.kinect_loader import _get_config, _get_device_ids
        from calib.utils import kinect_transform_mat

        if root_path is None:
            root_path = self.input_path

        self.started_devices = {}

        device_ids = _get_device_ids()

        for device_name in self.devices:
            self.started_devices[device_name] = PyK4A(config=_get_config(self.devices_type[device_name]), device_id=device_ids[self.devices_id[device_name]])
            self.started_devices[device_name].open()
            self.started_devices[device_name].start()

        def process(device, R, t):
            capture = device.get_capture()
            color_frame, pcl_frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB), capture.transformed_depth_point_cloud
            pcl = pcl_frame.reshape(-1,3)/1000 @ R.T + t
            colors = color_frame.reshape(-1, 3)/255
            return dict(pcl=pcl, colors=colors)

        transform_mats = kinect_transform_mat(root_path)

        pool = Pool()
        
        while True:
            results = {}
            for device_name in self.devices:
                R = transform_mats["kinect_{}".format(device_name)]["R"]
                t = transform_mats["kinect_{}".format(device_name)]["t"]
                results[device_name] = pool.apply_async(process, (self.started_devices[device_name], R, t)).get()
            yield results


class KinectOfflineStreamPlotCpp(O3DStreamPlot):
    def __init__(self, input_path: str, devices=['master','sub1','sub2'], start_frame=30, tag="st", *args, **kwargs) -> None:
        from kinect.config import MAS, SUB1, SUB2, INTRINSIC
        self.input_path = input_path
        self.devices = devices
        self.devices_type = dict(master="mas", sub1="sub", sub2="sub")
        self.devices_id = dict(master=MAS, sub1=SUB1, sub2=SUB2)
        self.start_frame = start_frame
        self.tag = tag
        super().__init__(*args, **kwargs)

    def init_updater(self):
        self.plot_funcs = {}
        for device_name in self.devices:
            self.plot_funcs[device_name] = o3d_pcl

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([[0],[0],[1]]))
        self.ctr.set_front(np.array([[0],[-1],[0]]))
        self.ctr.set_zoom(1)

    def generator(self, root_path: str = None):
        import cv2
        from run_calibration import run_kinect_calib_cpp
        from dataloader.result_loader import KinectResultLoader

        if root_path is None:
            root_path = self.input_path

        transform_mats = run_kinect_calib_cpp(root_path, self.devices)

        if self.tag == "id":
            params = []
            for device in self.devices:
                params.append(dict(tag="kinect/{}/pcls".format(device), ext=".npy"))
                params.append(dict(tag="kinect/{}/color".format(device), ext=".png"))
            loader = KinectResultLoader(root_path, params)
            for v in range(self.start_frame, len(loader)):
                result = {}
                pcl_frame = loader[v]
                for dev in self.devices:
                    pcl = np.load(pcl_frame["kinect/{}/pcls".format(dev)]["filepath"]).reshape(-1,3)/1000
                    color = cv2.imread(pcl_frame["kinect/{}/color".format(dev)]["filepath"])
                    result[dev] = o3d_pcl(pcl, colors=np.fliplr(color.reshape(-1, 3)/255)).transform(transform_mats[dev])
                yield result

        else:
            pcl_params = []
            color_params = []
            for device in self.devices:
                pcl_params.append(dict(tag="kinect/{}/pcls".format(device), ext=".npy"))
                color_params.append(dict(tag="kinect/{}/color".format(device), ext=".png"))
            pcl_loader = KinectResultLoader(root_path, pcl_params)
            color_loader = KinectResultLoader(root_path, color_params)
            for i in range(self.start_frame, len(pcl_loader)):
            # for v in pcl_loader.file_dict["kinect/master/pcls"].loc[self.start_frame:, "st"]:
                pcl_frame = pcl_loader.select_item(pcl_loader[i]["kinect/master/pcls"]["st"], "st", False)
                result = {}

                for dev in self.devices:
                    pcl = np.load(pcl_frame["kinect/{}/pcls".format(dev)]["filepath"]).reshape(-1,3)/1000
                    color_frame = color_loader.select_by_id(pcl_frame["kinect/{}/pcls".format(dev)]["id"])
                    color = cv2.imread(color_frame["kinect/{}/color".format(dev)]["filepath"])
                    result[dev] = o3d_pcl(pcl, colors=np.fliplr(color.reshape(-1, 3)/255)).transform(transform_mats[dev])
                    # o3d_pcls = o3d_pcl(pcl, colors=np.fliplr(color.reshape(-1, 3)/255)).transform(transform_mats[device])
                    # result[device] = dict(pcl=np.asarray(o3d_pcls.points), colors=np.asarray(o3d_pcls.colors))
                yield result

    def show(self):
        gen = self.generator()
        for update_dict in gen:
            o3d_plot(list(update_dict.values()))


class OptiArbeManager():
    def __init__(self, result_path) -> None:
        self.opti_loader = OptitrackResultLoader(result_path)
        self.arbe_loader = ArbeResultLoader(result_path)

    def generator(self):
        for i in range(30, len(self.arbe_loader)):
            arbe_row = self.arbe_loader[i]
            opti_row = self.opti_loader.select_item(arbe_row["arbe"]["st"], "st", False)
            yield opti_row["optitrack"], arbe_row["arbe"]


class OptitrackArbeStreamPlot(O3DStreamPlot):
    def __init__(self, input_path: str, angle_of_view=[0,-1,0,2], *args, **kwargs) -> None:
        super().__init__(width=800, *args, **kwargs)
        self.input_path = input_path
        self.angle_of_view = angle_of_view

    def init_updater(self):
        self.plot_funcs = dict(
            opti_marker=o3d_skeleton,
            # opti_bone=o3d_pcl,
            arbe_pcls=o3d_pcl)

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([0, 0, 1]))
        self.ctr.set_front(np.array(self.angle_of_view[:3]))
        self.ctr.set_zoom(self.angle_of_view[3])
        
    def generator(self):
        from calib.utils import optitrack_transform_mat
        from optitrack.config import marker_lines
        input_manager = OptiArbeManager(self.input_path)
        transform_mats = optitrack_transform_mat(self.input_path)

        R_opti_T = transform_mats["optitrack"]["R"].T
        t_opti = transform_mats["optitrack"]["t"]

        for opti_row, arbe_row in input_manager.generator():
            # load numpy from file
            opti_arr_dict = np.load(opti_row["filepath"])
            arbe_arr = np.load(arbe_row["filepath"])

            person_count = opti_arr_dict["markers"].shape[0]
            markers_pcl = opti_arr_dict["markers"][:,:,:3].reshape(-1,3)
            # bones_pcl = opti_arr_dict["bones"][:,:,:3].reshape(-1,3)

            # transform
            opti_markers = markers_pcl @ R_opti_T + t_opti
            # opti_bones = bones_pcl @ R_opti_T + t_opti

            # filter pcl with naive bounding box
            arbe_pcl = pcl_filter(opti_markers, arbe_arr[:,:3], 0.5)

            # init lines
            lines = marker_lines
            if person_count > 1:
                for p in range(1, person_count):
                    lines = np.vstack((lines, lines + p*37))
            colors = np.asarray([[0,0,1]] * len(lines))
            
            yield dict(
                opti_marker=dict(
                    skeleton=opti_markers,
                    color=[0,0,1],
                    lines=lines,
                    colors=colors
                ),
                # opti_bone=dict(
                #     pcl=opti_bones,
                #     color=[1,0,0]
                # ),
                arbe_pcls=dict(
                    pcl=arbe_pcl,
                    color=[0,1,0]
                ),
            )
