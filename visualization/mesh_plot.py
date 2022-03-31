import time
from typing import Generator, Tuple
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
            opti=o3d_pcl,
            skel=o3d_skeleton,
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


class MinimalResultParallelPlot(O3DStreamPlot):
    save_path = ''
    updater_dict = {}
    def __init__(self, root_path, *args, **kwargs) -> None:
        from dataloader.result_loader import MinimalLoader
        from minimal import config
        from minimal.models import KinematicModel, KinematicPCAWrapper
        import os
        import json

        self.skip_head = int(kwargs.pop("skip_head", 0))
        self.skip_tail = int(kwargs.pop("skip_tail", 0))
        self._pause = bool(kwargs.pop("pause", True))
        MinimalResultParallelPlot.save_path = kwargs.pop("save_path", "./")

        super().__init__(width=1800, *args, **kwargs)

        self.timer = Timer()

        self.o_loader = MinimalLoader(root_path, [dict(tag="minimal/param", ext=".npz")])
        self.k_loader = MinimalLoader(root_path, [dict(tag="minimal_k/param", ext=".npz")])

        model_path = config.SMPL_MODEL_1_0_PATH
        with open(os.path.join(root_path, "infomation.json"), "r") as f:
            record_info = json.load(f)
            if record_info.get(" gender", None) == "male" or record_info.get("gender", None) == "male":
                model_path = config.SMPL_MODEL_1_0_MALE_PATH

        self.smpl = KinematicPCAWrapper(KinematicModel().init_from_file(model_path))

        def save(v):
            import open3d as o3d

            save_path = MinimalResultParallelPlot.save_path

            o3d.io.write_triangle_mesh(os.path.join(save_path, "o_mesh.ply"), MinimalResultParallelPlot.updater_dict["o_mesh"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "k_mesh.ply"), MinimalResultParallelPlot.updater_dict["k_mesh"].update_item)

        self.view.register_key_callback(66, save)

    def init_updater(self):
        self.plot_funcs = dict(
            o_mesh=o3d_mesh,
            k_mesh=o3d_mesh,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([0, 0, 1]))
        self.ctr.set_front(np.array([0, -1, 0]))
        self.ctr.set_zoom(0.6)

    def generator(self):
        MinimalResultParallelPlot.updater_dict = self.updater_dict
        init_face = True

        for i in range(self.skip_head, min(len(self.o_loader), len(self.k_loader))-self.skip_tail):
            o_param = np.load(self.o_loader[i]["minimal/param"]["filepath"])
            k_param = np.load(self.k_loader[i]["minimal_k/param"]["filepath"])

            print(i)
            # if self._pause:
            #     self.pause = self._pause

            o_pose = o_param["pose"]
            o_pose[3+22*3:3+24*3] = 0
            k_pose = k_param["pose"]
            k_pose[3+22*3:3+24*3] = 0
            self.smpl.run(np.hstack((o_pose, o_param["shape"])))
            o_mesh = (self.smpl.core.verts, self.smpl.core.faces)
            self.smpl.run(np.hstack((k_pose, o_param["shape"])))
            k_mesh = (self.smpl.core.verts, self.smpl.core.faces)
            
            yield dict(
                o_mesh=dict(mesh=o_mesh, color=np.array([165, 222, 229])/255),
                k_mesh=dict(mesh=k_mesh, color=np.array([247, 218, 113])/255),
                # k_mesh=dict(mesh=(self.smpl.core.verts, self.smpl.core.faces), color=[0.2, 1, 0.2]),
            )


class MinimalResultStreamPlot(O3DStreamPlot):
    def __init__(self, root_path, *args, **kwargs) -> None:
        skip_head = int(kwargs.pop("skip_head", 0))
        skip_tail = int(kwargs.pop("skip_tail", 0))
        self.show_rgb = kwargs.pop("show_rgb", False)

        super().__init__(width=1800, *args, **kwargs)

        sources = ["mesh", "mesh_param", "master", "kinect_pcl", "kinect_pcl_remove_zeros"]
        if self.show_rgb:
            sources.append("kinect_color")
        self.file_loader = ResultFileLoader(root_path, skip_head, skip_tail, enabled_sources=sources)
        self.timer = Timer()

    def init_updater(self):
        self.plot_funcs = dict(
            mesh=o3d_mesh,
            # kpts=o3d_skeleton,
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
            vertices = frame["mesh_param"]["vertices"]
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
                # kpts=dict(skeleton=frame["optitrack"], lines=marker_lines, color=[1,0,0]),
                pcl=dict(pcl=pcl_filter(vertices, frame["master_pcl"])),
            )


class MeshPclStreamPlot(O3DStreamPlot):
    updater_dict = {}
    info = {}
    save_path = ''
    def __init__(self, root_path, *args, **kwargs) -> None:
        skip_head = int(kwargs.pop("skip_head", 0))
        skip_tail = int(kwargs.pop("skip_tail", 0))
        MeshPclStreamPlot.save_path = kwargs.pop("save_path", "./")

        super().__init__(width=1800, *args, **kwargs)

        sources = ["mesh", "mesh_param", "arbe", "arbe_pcl"]
        self.file_loader = ResultFileLoader(root_path, skip_head, skip_tail, enabled_sources=sources)
        self.timer = Timer()

        self.info = {}

        def save(v):
            import open3d as o3d
            import os

            save_path = MeshPclStreamPlot.save_path

            o3d.io.write_triangle_mesh(os.path.join(save_path, "radar_id={}.ply".format(MeshPclStreamPlot.info["arbe"]["id"])), MeshPclStreamPlot.updater_dict["pcl"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "mesh_id={}.ply".format(MeshPclStreamPlot.info["mesh"]["id"])), MeshPclStreamPlot.updater_dict["mesh"].update_item)

        self.view.register_key_callback(66, save)


    def init_updater(self):
        self.plot_funcs = dict(
            mesh=o3d_mesh,
            # kpts=o3d_skeleton,
            pcl=o3d_mesh,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([0, 0, 1]))
        self.ctr.set_front(np.array([0, -1, 0]))
        self.ctr.set_zoom(0.2)

    def generator(self):
        from minimal.models import KinematicModel, KinematicPCAWrapper
        from minimal.config import SMPL_MODEL_1_0_PATH, SMPL_MODEL_1_0_MALE_PATH
        MeshPclStreamPlot.updater_dict = self.updater_dict

        gender = self.file_loader.info_dict["gender"]
        smpl_model = KinematicPCAWrapper(KinematicModel().init_from_file(SMPL_MODEL_1_0_MALE_PATH if gender == "male" else SMPL_MODEL_1_0_PATH, compute_mesh=False))

        for i in range(len(self.file_loader)):
            # self.pause = True

            frame, info = self.file_loader[i]
            pose = frame["mesh_param"]["pose"]
            pose[3+22*3:3+24*3] = 0
            shape = frame["mesh_param"]["shape"]
            
            smpl_model.run(np.hstack((pose, shape)))

            # print(1/self.timer.tic())
            MeshPclStreamPlot.info["mesh"] = info["mesh_param"]
            MeshPclStreamPlot.info["arbe"] = info["arbe"]
            print(MeshPclStreamPlot.info)
            
            yield dict(
                mesh=dict(mesh=(smpl_model.core.verts, smpl_model.core.faces)),
                # kpts=dict(skeleton=frame["optitrack"], lines=marker_lines, color=[1,0,0]),
                pcl=dict(mesh=pcl2sphere(pcl_filter(smpl_model.core.keypoints, frame["arbe"])), color=[0.2,1,0.2]),
            )


class MeshEvaluateStreamPlot(O3DStreamPlot):
    updater_dict = {}
    save_path = ''
    def __init__(self, *args, **kwargs) -> None:
        MeshEvaluateStreamPlot.save_path = kwargs.pop("save_path", "./")
        super().__init__(width=1800, *args, **kwargs)

        MeshEvaluateStreamPlot.updater_dict = self.updater_dict
        self.idx = 0

        def save(v):
            import open3d as o3d
            import os

            save_path = MeshEvaluateStreamPlot.save_path

            o3d.io.write_triangle_mesh(os.path.join(save_path, "radar_pcl{}.ply".format(self.idx)), MeshEvaluateStreamPlot.updater_dict["radar_pcl"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "pred_smpl{}.ply".format(self.idx)), MeshEvaluateStreamPlot.updater_dict["pred_smpl"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "label_smpl{}.ply".format(self.idx)), MeshEvaluateStreamPlot.updater_dict["label_smpl"].update_item)
            self.idx += 1
        self.view.register_key_callback(66, save)

    def init_updater(self):
        self.plot_funcs = dict(
            radar_pcl=o3d_mesh,
            pred_smpl=o3d_smpl_mesh,
            label_smpl=o3d_smpl_mesh,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([[0],[0],[1]]))
        self.ctr.set_front(np.array([[0],[-1],[0]]))
        self.ctr.set_lookat(np.array([0,0,0]))
        self.ctr.set_zoom(1)

class MoshEvaluateStreamPlot(O3DStreamPlot):
    updater_dict = {}
    save_path = ''
    def __init__(self, *args, **kwargs) -> None:
        MoshEvaluateStreamPlot.save_path = kwargs.pop("save_path", "./")
        super().__init__(width=1800, *args, **kwargs)

        MoshEvaluateStreamPlot.updater_dict = self.updater_dict
        self.idx = 0

        def save(v):
            import open3d as o3d
            import os

            save_path = MoshEvaluateStreamPlot.save_path

            o3d.io.write_triangle_mesh(os.path.join(save_path, "radar_pcl{}.ply".format(self.idx)), MoshEvaluateStreamPlot.updater_dict["radar_pcl"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "pred_smpl{}.ply".format(self.idx)), MoshEvaluateStreamPlot.updater_dict["pred_smpl"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "label_smpl{}.ply".format(self.idx)), MoshEvaluateStreamPlot.updater_dict["label_smpl"].update_item)
            self.idx += 1
        self.view.register_key_callback(66, save)

    def init_updater(self):
        self.plot_funcs = dict(
            radar_pcl=o3d_mesh,
            pred_smpl=o3d_mesh,
            label_smpl=o3d_mesh,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([[0],[0],[1]]))
        self.ctr.set_front(np.array([[0],[-1],[0]]))
        self.ctr.set_lookat(np.array([0,0,0]))
        self.ctr.set_zoom(1)


def pcl2sphere(pcl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    faces = []
    r = 0.0075
    curr_faces = [
        [0,1,2],[1,3,2],[0,4,1],[1,4,5],[4,6,5],[5,6,7],[2,3,6],[3,7,6],[0,2,4],[2,6,4],[1,5,3],[3,5,7]
    ]
    for i in range(pcl.shape[0]):
        for u in [-1, 1]:
            for v in [-1, 1]:
                for w in [-1, 1]:
                    verts.append(pcl[i] + np.array([u*r, v*r, w*r]))
        verts_idxs = np.array(range(i*8, i*8+8))
        
        for face in curr_faces:
            faces.append(verts_idxs[face])

    return np.array(verts, dtype=float), np.array(faces, dtype=int)
