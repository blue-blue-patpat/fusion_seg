from visualization.utils import O3DStreamPlot, o3d_coord, o3d_mesh, o3d_pcl, o3d_plot, o3d_lines, filter_pcl
import numpy as np
import open3d as o3d
from dataloader.result_loader import ArbeResultLoader


class ArbeManager():
    def __init__(self, result_path) -> None:
        self.arbe_loader = ArbeResultLoader(result_path)

    def generator(self):
        for i in range(0, len(self.arbe_loader)):
            arbe_row = self.arbe_loader[i]
            print(arbe_row["arbe"])
            if arbe_row["arbe"]:
                yield arbe_row["arbe"]


class ArbeStreamPlot(O3DStreamPlot):
    def __init__(self, input_path: str, angle_of_view=[0, -1, 0, 1], *args, **kwargs) -> None:
        super().__init__(width=800, *args, **kwargs)
        self.input_path = input_path
        self.angle_of_view = angle_of_view

    def init_updater(self):
        self.plot_funcs = dict(
            arbe_pcls=o3d_pcl,
        )

    def init_show(self):
        super().init_show()
        self.ctr.set_up(np.array([0, 0, 1]))
        self.ctr.set_front(np.array(self.angle_of_view[:3]))
        self.ctr.set_zoom(self.angle_of_view[3])

    def generator(self):
        input_manager = ArbeManager(self.input_path)

        for arbe_row in input_manager.generator():
            # load numpy from file
            arbe_arr = np.load(arbe_row["filepath"])

            yield dict(
                arbe_pcls=dict(
                    pcl=arbe_arr[:, :3],
                    color=[0, 1, 0]
                ),
            )


"""
Visualize optitrack skeleton and arbe pcl
"""


# root_path = "/home/nesc525/drivers/5/mmvoice/2022-08-06_15-58-37"
# plot = ArbeStreamPlot(root_path)

# arbe_loader = ArbeResultLoader(root_path)

# for i in range(0, len(arbe_loader)):
#     arbe_row = arbe_loader[i]
#     print(arbe_row["arbe"])
#     if arbe_row["arbe"]:
#         arbe_arr = np.load(arbe_row["arbe"]["filepath"])
#         arbe_arr = pcl_filter(np.array([[-0.2, 1.5, -0.2], [0.2, 2.5, 0.2]]), arbe_arr, 0)

#         rate = (arbe_arr[:, 7] + 0.3)/0.6
#         # input()

#         o3d_plot([o3d_pcl(pcl=arbe_arr[:,:3] - np.array([0,2,0]),
#                           colors=(np.asarray([[1, 0, 0]] * rate.shape[0]).T * (1-rate) + np.asarray([[0, 1, 0]] * rate.shape[0]).T * rate).T)])


import librosa
import soundfile as sf
import wave
import os
root_path = "/home/nesc525/drivers/5/mmvoice/2022-08-06_15-58-37"
wave_path = os.path.join(root_path, 'audio/id=0_st=1659772723.3297486.wav')
x,_ = librosa.load(wave_path)
sf.write('tmp.wav', x)
wave.open('tmp.wav','r')