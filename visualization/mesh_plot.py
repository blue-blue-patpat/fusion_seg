import numpy as np
import cv2
from pytorch3d.io.obj_io import load_obj
from vctoolkit import Timer

from dataloader.result_loader import ResultFileLoader
from visualization.utils import O3DStreamPlot, o3d_coord, o3d_mesh, o3d_pcl, o3d_plot, o3d_skeleton, o3d_smpl_mesh, pcl_filter


class MinimalStreamPlot(O3DStreamPlot):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(width=1800, *args, **kwargs)

    def init_updater(self):
        self.plot_funcs = dict(
            mesh=o3d_mesh,
            pcl=o3d_pcl,
            kpts=o3d_pcl,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([[0],[0],[1]]))
        self.ctr.set_front(np.array([[0.2],[-1],[0]]))
        self.ctr.set_lookat(np.array([0,4,0]))
        self.ctr.set_zoom(0.6)


class MinimalInputStreamPlot(O3DStreamPlot):
    def __init__(self, root_path, skip_head, skip_tail, *args, **kwargs) -> None:
        super().__init__(width=1800, *args, **kwargs)
        sources = ["optitrack", "sub1", "kinect_pcl", "kinect_pcl_remove_zeros", "kinect_skeleton"]
        self.file_loader = ResultFileLoader(root_path, skip_head, skip_tail,  enabled_sources=sources)
        self.timer = Timer()

    def init_updater(self):
        self.plot_funcs = dict(
            opti_skeleton=o3d_skeleton,
            kinect_skeleton=o3d_skeleton,
            pcl=o3d_pcl,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([[0],[0],[1]]))
        self.ctr.set_front(np.array([[0.2],[-1],[0]]))
        self.ctr.set_lookat(np.array([0,4,0]))
        self.ctr.set_zoom(0.6)

    def generator(self):
        from minimal.utils import SMPL_SKELETON_LINES
        from minimal.bridge import JointsBridge

        bridge = JointsBridge()

        for i in range(len(self.file_loader)):
            frame, info = self.file_loader[i]

            pcl = frame["sub1_pcl"]

            bridge.init_input(frame["optitrack"], pcl)
            opti_jnts, _pcl = bridge.map()

            bridge.init_input(frame["sub1_skeleton"], _pcl)
            kinect_jnts, _pcl = bridge.map("kinect", use_filter=False)
            print(1/self.timer.tic())
            
            yield dict(
                opti_skeleton=dict(skeleton=opti_jnts, lines=SMPL_SKELETON_LINES, color=[1,0,0]),
                kinect_skeleton=dict(skeleton=kinect_jnts, lines=SMPL_SKELETON_LINES, color=[0,0,1]),
                pcl=dict(pcl=_pcl, color=[0,0.6,0]),
            )


class MinimalResultStreamPlot(O3DStreamPlot):
    def __init__(self, root_path, *args, **kwargs) -> None:
        skip_head = int(kwargs.pop("skip_head", 0))
        skip_tail = int(kwargs.pop("skip_tail", 0))
        self.show_rgb = kwargs.pop("show_rgb", False)

        super().__init__(width=1800, *args, **kwargs)

        sources = ["mesh", "mesh_param", "optitrack", "master", "kinect_pcl", "kinect_pcl_remove_zeros"]
        if self.show_rgb:
            sources.append("kinect_color")
        self.file_loader = ResultFileLoader(root_path, skip_head, skip_tail, enabled_sources=sources)
        self.timer = Timer()

    def init_updater(self):
        self.plot_funcs = dict(
            mesh=o3d_mesh,
            kpts=o3d_skeleton,
            pcl=o3d_pcl,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([0, 0, 1]))
        self.ctr.set_front(np.array([0, -1, 0]))
        self.ctr.set_zoom(0.2)
        if self.show_rgb:
            cv2.namedWindow('Mesh RGB',0)

    def update_plot(self):
        if self.show_rgb:
            cv2.imshow("Mesh RGB", cv2.resize(self.img, (800, 600)))
            cv2.waitKey(10)
        return super().update_plot()

    def generator(self):
        from optitrack.config import marker_lines

        init_faces = True

        for i in range(len(self.file_loader)):
            frame, info = self.file_loader[i]
            vertices = (frame["mesh_param"]["vertices"] @ frame["mesh_R"] + frame["mesh_t"]) * frame["mesh_scale"]
            faces = None
            if init_faces:
                _v, faces, _t = load_obj(self.file_loader.mesh_loader.file_dict["minimal/obj"].iloc[0]["filepath"])
                faces = faces[0]
                init_faces = False
            print(1/self.timer.tic())
            
            if self.show_rgb:
                self.img=frame["master_color"]
            
            yield dict(
                mesh=dict(mesh=(vertices, faces)),
                kpts=dict(skeleton=frame["optitrack"], lines=marker_lines, color=[1,0,0]),
                pcl=dict(pcl=pcl_filter(frame["optitrack"], frame["master_pcl"][np.random.choice(np.arange(frame["master_pcl"].shape[0]), size=5000, replace=False)])),
            )


class MeshEvaluateStreamPlot(O3DStreamPlot):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(width=1800, *args, **kwargs)

    def init_updater(self):
        self.plot_funcs = dict(
            radar_pcl=o3d_pcl,
            pred_smpl=o3d_smpl_mesh,
            label_smpl=o3d_smpl_mesh,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([[0],[0],[1]]))
        self.ctr.set_front(np.array([[0],[-1],[0]]))
        self.ctr.set_lookat(np.array([0,0,0]))
        self.ctr.set_zoom(1)
