import numpy as np
import open3d as o3d

from dataloader.result_loader import ResultFileLoader
from visualization.utils import O3DStreamPlot, o3d_coord, o3d_mesh, o3d_pcl, o3d_plot, o3d_skeleton, pcl_filter


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
        self.ctr.set_up(np.array([[0],[1],[0]]))
        self.ctr.set_front(np.array([[0.2],[0],[1]]))
        self.ctr.set_zoom(0.6)


class MinimalResultStreamPlot(O3DStreamPlot):
    def __init__(self, root_path, *args, **kwargs) -> None:
        skip_head = int(kwargs.pop("skip_head", 0))
        skip_tail = int(kwargs.pop("skip_tail", 0))
        super().__init__(width=1800, *args, **kwargs)
        self.file_loader = ResultFileLoader(root_path, skip_head, skip_tail,
            enabled_sources=["mesh", "mesh_obj", "optitrack", "master", "sub1", "sub2", "kinect_pcl", "kinect_pcl_remove_zeros"])

    def init_updater(self):
        self.plot_funcs = dict(
            mesh=o3d_mesh,
            kpts=o3d_skeleton,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([0, 0, 1]))
        self.ctr.set_front(np.array([-1, -1, 0]))
        self.ctr.set_zoom(0.4)

    def generator(self):
        from optitrack.config import marker_lines

        for i in range(len(self.file_loader)):
            frame, info = self.file_loader[i]

            yield dict(
                mesh=dict(mesh=frame["mesh_obj"]),
                kpts=dict(skeleton=frame["optitrack"], lines=marker_lines, color=[1,0,0])
            )
