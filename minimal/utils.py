from typing import Tuple, List, Dict, Optional, Callable
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot, ExtremaPrinter
from livelossplot.main_logger import MainLogger, LogItem


class ExtremaPrinterWithExclude(ExtremaPrinter):
  def __init__(self,
        massage_template: str = '\t{metric_name:16} \t (min: {min:8.3f},'
        ' max: {max:8.3f}, cur: {current:8.3f})'):
        super().__init__(massage_template=massage_template)

  def send(self, logger: MainLogger):
        log_groups = logger.grouped_log_history()
        log_groups.pop("_exclude")
        self.last_message = '\n'.join(self._create_massages(log_groups))
        print(self.last_message)


class MeshPlot(MatplotlibPlot):
    def __init__(
        self,
        cell_size: Tuple[int, int] = (6, 4),
        max_cols: int = 2,
        max_epoch: int = None,
        skip_first: int = 2,
        extra_plots: List[Callable[[MainLogger], None]] = [],
        figpath: Optional[str] = None,
        after_subplot: Optional[Callable[[plt.Axes, str, str], None]] = None,
        before_plots: Optional[Callable[[plt.Figure, np.ndarray, int], None]] = None,
        after_plots: Optional[Callable[[plt.Figure], None]] = None,
    ):
        super().__init__(cell_size=cell_size, max_cols=max_cols, max_epoch=max_epoch, skip_first=skip_first, extra_plots=extra_plots, figpath=figpath, after_subplot=after_subplot, before_plots=before_plots, after_plots=after_plots)

    def send(self, logger: MainLogger):

        """Draw figures with metrics and show"""
        log_groups = logger.grouped_log_history()

        max_rows = math.ceil((len(log_groups) + len(self.extra_plots)) / self.max_cols)

        fig = plt.figure()
        # axes = axes.reshape(-1, self.max_cols)
        axes = np.array([])
        self._before_plots(fig, axes, len(log_groups))
        for group_idx, (group_name, group_logs) in enumerate(log_groups.items()):
            # ax = axes[group_idx // self.max_cols, group_idx % self.max_cols]
            if group_name == "_exclude":
                self._mesh_plot(fig, group_logs, 1, 75)
                self._mesh_plot(fig, group_logs, 2, -75)
            elif any(len(logs) > 0 for name, logs in group_logs.items()):
                ax = fig.add_subplot(1, 3, 3)
                self._draw_metric_subplot(ax, group_logs, group_name=group_name, x_label=logger.step_names[group_name])

        self._after_plots(fig)

    def _mesh_plot(self, fig, logger, idx=1, azim=75):
        mesh = logger.get("mesh", [None])
        pcls = logger.get("pcls", [None])
        kpts_target = logger.get("kpts_target", [None])
        kpts_updated = logger.get("kpts_updated", [None])
        if None in mesh[-1].value:
            return
        x, y, z = mesh[-1].value[0].T
        # ax = Axes3D(fig)
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        if not None in pcls[-1].value:
            pcl_x, pxl_y, pxl_z = pcls[-1].value.T
            ax.scatter(pcl_x, -pxl_z, pxl_y, color=[(0.62,0.92,0.42,0.3)], depthshade=False)
        if not None in kpts_target[-1].value:
            kpts_target_x, kpts_target_y, kpts_target_z = kpts_target[-1].value.T
            ax.scatter(kpts_target_x, -kpts_target_z, kpts_target_y, color=[(0,1,0,1)], depthshade=False)
        if not None in kpts_updated[-1].value:
            kpts_updated_x, kpts_updated_y, kpts_updated_z = kpts_updated[-1].value.T
            ax.scatter(kpts_updated_x, -kpts_updated_z, kpts_updated_y, color=[(0,0,1,1)], depthshade=False)
        ax.plot_trisurf(x, -z, y, triangles=mesh[-1].value[1], cmap=plt.cm.Spectral)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title("mesh")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        # 仰角 方位角
        ax.view_init(elev=0, azim=azim)


class WeightLoss(list):
    def __init__(self, weight=1) -> None:
        super().__init__()
        self.weight = weight

    def delta(self, idx=-1, absolute=True):
        if absolute:
            return abs(self[idx-1] - self[idx])
        else:
            return self[idx-1] - self[idx]
    
    def __getitem__(self, i):
        try:
            ret = super(WeightLoss, self).__getitem__(i) * self.weight
        except:
            ret = 0.
        return ret

class LossManager():
    def __init__(self, losses_with_weights={}, mse_threshold=1e-8, loss_threshold=1e-8) -> None:
        self.losses = {}
        self.epoch = 0
        self.mse_threshold = mse_threshold
        self.loss_threshold = loss_threshold
        self.add_loss(losses_with_weights)

        self.plotlosses = PlotLosses(groups={'losses': list(self.losses.keys())+['loss'], '_exclude': ['mesh', 'pcls', 'kpts_target', 'kpts_updated']},
                                    outputs=[MeshPlot(cell_size=[10, 3]), ExtremaPrinterWithExclude()])

    def add_loss(self, losses_with_weights: dict):
        for loss_name, weight in losses_with_weights.items():
            self.losses[loss_name] = WeightLoss(weight)

    def update_loss(self, loss_name: str, value: float):
        if loss_name in self.losses.keys():
            self.losses[loss_name].append(value)

    def update_losses(self, losses_value: dict):
        for loss_name, value in losses_value.items():
            self.update_loss(loss_name, value)
        self.epoch += 1

    def delta(self, idx=-1, absolute=True):
        return sum([loss.delta(idx, absolute) for loss in self.losses.values()])

    def check_losses(self):
        return self[-1] < self.loss_threshold or self.delta() < self.mse_threshold

    def str_losses(self, idx=-1):
        return "\t".join(["{}={:.4f}".format(loss_name, loss[idx]) for loss_name, loss in self.losses.items()])

    def show_losses(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 10))
        ax = fig.gca()
        for loss_name, loss in self.losses: 
            ax.plot(loss, label=loss_name)
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs iterations", fontsize="16")
        plt.show()
        return fig

    def update_loss_curve(self, **kwargs):
        log = dict(zip(self.losses.keys(), [loss[-1] for loss in self.losses.values()]))
        log["loss"] = self[-1]
        log["mesh"] = kwargs.get("mesh", None)
        log["pcls"] = kwargs.get("pcls", None)
        log["kpts_target"] = kwargs.get("kpts_target", None)
        log["kpts_updated"] = kwargs.get("kpts_updated", None)
        self.plotlosses.update(log)
        self.plotlosses.send()

    def __len__(self):
        return min([len(loss) for loss in self.losses.values()])

    def __getitem__(self, i):
        return sum([loss[i] for loss in self.losses.values()])
