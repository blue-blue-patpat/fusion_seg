from typing import Tuple, List, Dict, Optional, Callable
import math
import os

from minimal.models import KinematicModel, KinematicPCAWrapper
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot, ExtremaPrinter
from livelossplot.main_logger import MainLogger, LogItem
import numpy as np
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt


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

        fig, axes = plt.subplots(max_rows, self.max_cols)
        axes = axes.reshape(-1, self.max_cols)
        self._before_plots(fig, axes, len(log_groups))

        for group_idx, (group_name, group_logs) in enumerate(log_groups.items()):
            ax = axes[group_idx // self.max_cols, group_idx % self.max_cols]
            if group_name == "_exclude":
                ax.set_axis_off()
                self._mesh_plot(fig, group_logs)
            elif any(len(logs) > 0 for name, logs in group_logs.items()):
                self._draw_metric_subplot(ax, group_logs, group_name=group_name, x_label=logger.step_names[group_name])

        for idx, extra_plot in enumerate(self.extra_plots):
            ax = axes[(len(log_groups) + idx) // self.max_cols, (len(log_groups) + idx) % self.max_cols]
            extra_plot(ax, logger)

        self._after_plots(fig)

    def _mesh_plot(self, fig, logger):
        mesh = logger.get("mesh", [None])
        if mesh[-1].value is None:
            return
        x, y, z = mesh[-1].value[0].T
        # ax = Axes3D(fig)
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_trisurf(x, -z, y, triangles=mesh[-1].value[1], cmap=plt.cm.Spectral)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title("mesh")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)


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

        self.plotlosses = PlotLosses(from_step=-5, groups={'losses': list(self.losses.keys())+['loss'], '_exclude': ['mesh']},
                                    outputs=[MeshPlot(cell_size=[10,5]), ExtremaPrinterWithExclude()])

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
        self.plotlosses.update(log)
        self.plotlosses.send()

    def __len__(self):
        return min([len(loss) for loss in self.losses.values()])

    def __getitem__(self, i):
        return sum([loss[i] for loss in self.losses.values()])


class Solver:
    def __init__(self, model: KinematicPCAWrapper, eps=1e-5, max_iter=30):
        """
        Parameters
        ----------
        eps : float, optional
        Epsilon for derivative computation, by default 1e-5
        max_iter : int, optional
        Max iterations, by default 30
        mse_threshold : float, optional
        Early top when mse change is smaller than this threshold, by default 1e-8
        verbose : bool, optional
        Print information in each iteration, by default False
        """
        self.model = model
        self.eps = eps
        self.max_iter = max_iter

        self.params = []

    def solve_full(self, kpts_target, pcls_target, init=None, losses_with_weights=None,
                    mse_threshold=1e-8, loss_threshold=1e-8, u=1e-3, v=1.5, verbose=False):
        if init is None:
            init = np.zeros(self.model.n_params)

        if losses_with_weights is None:
            losses_with_weights = dict(
                kpts_losses=1,
                # edge_losses=1,
                # face_losses=1,
            )

        out_n = np.shape(kpts_target.flatten())[0]
        jacobian = np.zeros([out_n, init.shape[0]])

        losses = LossManager(losses_with_weights, mse_threshold, loss_threshold)

        params = init

        for i in range(self.max_iter):
            # update modle
            mesh_updated, keypoints_updated = self.model.run(params)

            # compute keypoints loss
            residual = (keypoints_updated - kpts_target).reshape(out_n, 1)
            loss_kpts = np.mean(np.square(residual))
            losses.update_loss("kpts_losses", loss_kpts)

            # # compute edge loss
            # loss_edge = point_mesh_edge_distance(mesh_updated, pcls_target)
            # losses.update_loss("edge_losses", loss_edge)
        
            # # compute face loss
            # loss_face = point_mesh_face_distance(mesh_updated, pcls_target)
            # losses.update_loss("face_losses", loss_face)

            # check loss
            if losses.check_losses():
                break

            for k in range(params.shape[0]):
                jacobian[:, k] = np.hstack([self.get_derivative(params, k)])
            jtj = np.matmul(jacobian.T, jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            delta = np.matmul(
                np.matmul(np.linalg.inv(jtj), jacobian.T), residual
            ).ravel()
            params -= delta

            update = losses.delta(absolute=False)
            print(update)

            if update > 0 and update > losses.delta(idx=-2, absolute=False):
                u /= v
            else:
                u *= v

            if verbose:
                # pcls = sample_points_from_meshes(mesh_updated, 1000).clone().detach().cpu().squeeze().unbind(1)
                vertices, faces= mesh_updated.verts_packed(), mesh_updated.faces_packed()
                losses.update_loss_curve(mesh=(vertices, faces))
                # print("[{}] : idx={}\tperiod={:.2f}\tloss={:.4f}\t{}".format(ymdhms_time() , i, 0, losses[-1], losses.str_losses(-1)))

        self.pose_params, self.shape_params = params[:-self.model.n_shape], params[-self.model.n_shape:]
        self.init_params = params
        return params

    def solve_pose(self, kpts_target, pcls_target, losses_with_weights=None,
                    mse_threshold=1e-8, loss_threshold=1e-8, u=1e-3, v=1.5, verbose=False):
        if losses_with_weights is None:
            losses_with_weights = dict(
                kpts_losses=1,
                # edge_losses=1,
                # face_losses=1,
            )

        out_n = np.shape(kpts_target.flatten())[0]
        jacobian = np.zeros([out_n, self.init_params.shape[0]])

        losses = LossManager(losses_with_weights, mse_threshold, loss_threshold)

        for i in range(self.max_iter):
            # update modle
            mesh_updated, keypoints_updated = self.model.run(np.hstack(self.pose_params, self.shape_params))

            # compute keypoints loss
            residual = (keypoints_updated - kpts_target).reshape(out_n, 1)
            loss_kpts = np.mean(np.square(residual))
            losses.update_loss("kpts_losses", loss_kpts)

            # # compute edge loss
            # loss_edge = point_mesh_edge_distance(mesh_updated, pcls_target)
            # losses.update_loss("edge_losses", loss_edge)
        
            # # compute face loss
            # loss_face = point_mesh_face_distance(mesh_updated, pcls_target)
            # losses.update_loss("face_losses", loss_face)

            # check loss
            if losses.check_losses():
                break

            for k in range(self.pose_params.shape[0]):
                jacobian[:, k] = np.hstack([self.get_derivative(self.pose_params, k)])
            jtj = np.matmul(jacobian.T, jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            delta = np.matmul(
                np.matmul(np.linalg.inv(jtj), jacobian.T), residual
            ).ravel()
            self.pose_params -= delta

            update = losses.delta(absolute=False)
            print(update)

            if update > 0 and update > losses.delta(idx=-2, absolute=False):
                u /= v
            else:
                u *= v

            if verbose:
                # pcls = sample_points_from_meshes(mesh_updated, 1000).clone().detach().cpu().squeeze().unbind(1)
                vertices, faces= mesh_updated.verts_packed(), mesh_updated.faces_packed()
                losses.update_loss_curve(mesh=(vertices, faces))
                # print("[{}] : idx={}\tperiod={:.2f}\tloss={:.4f}\t{}".format(ymdhms_time() , i, 0, losses[-1], losses.str_losses(-1)))
        return np.hstack(self.pose_params, self.shape_params)

    def solve_from_dir(self, input_path, output_path, device="master", vbs=[True, False]):
        from dataloader.result_loader import KinectResultLoader
        from minimal.bridge import JointsBridge
        pcls_tag = "kinect/{}/pcls".format(device)
        skeleton_tag = "kinect/{}/skeleton".format(device)
        T_tag = "kinect/{}/T".format(device)
        jnts_brg = JointsBridge()

        T_pcl = np.load(os.path.join(T_tag, "pcls.npy"))
        T_skeleton = np.load(os.path.join(T_tag, "skeleton.npy"))

        _jnts, _pcl = jnts_brg.smpl_from_kinect(T_skeleton[0], T_pcl)

        # solve init shape
        self.solve_full(_pcl, _jnts, verbose=vbs[0])

        rl = KinectResultLoader(input_path,
                                params=[dict(tag=pcls_tag, ext=".npy"),
                                        dict(tag=skeleton_tag, ext=".npy")])
        for item in rl.generator_by_skeleton():
            pcl = np.load(item[pcls_tag]["filepath"])
            skeleton = np.load(item[skeleton_tag]["filepath"])

            # only process skeleton 0
            _jnts, _pcl = jnts_brg.smpl_from_kinect(skeleton[0], pcl)

            params = self.solve_pose(_pcl, _jnts, verbose=vbs[1])
            info_source = item[skeleton_tag]
            np.save(os.path.join(output_path, "id={}_st={}_dt={}.npy".format(info_source["id"], info_source["st"], info_source["dt"])))


    def get_derivative(self, params, n):
        """
        Compute the derivative by adding and subtracting epsilon

        Parameters
        ----------
        model : object
        Model wrapper to be manipulated.
        params : np.ndarray
        Current model parameters.
        n : int
        The index of parameter.

        Returns
        -------
        np.ndarray
        Derivative with respect to the n-th parameter.
        """
        params1 = np.array(params)
        params2 = np.array(params)

        params1[n] += self.eps
        params2[n] -= self.eps

        res1 = self.model.run(params1)[1]
        res2 = self.model.run(params2)[1]

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()
