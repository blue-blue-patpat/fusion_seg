import os
from typing import Union
from visualization.utils import o3d_mesh, o3d_plot
import numpy as np
import torch
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from alfred.dl.torch.common import device
from minimal.models import KinematicModel, KinematicPCAWrapper
from minimal.utils import LossManager
from minimal.config import VPOSER_DIR
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser


class Solver:
    def __init__(self, model: KinematicPCAWrapper, eps=1e-5, plot_type="matplotlib"):
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
        self.plot_type = plot_type

        # coord_origin + pose_params
        self.pose_params = np.zeros(self.model.n_pose + 3)
        self.shape_params = np.zeros(self.model.n_shape)

        self.vp, _ = load_model(VPOSER_DIR, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        self.vp = self.vp.to('cuda')

    def solve(self, jnts_target, pcls_target, solve_type="full", losses_with_weights=None, max_iter=30,
                    kpts_threshold=0.04, mse_threshold=1e-7, loss_threshold=1e-6, u=1e-3, v=1.5, dbg_level=0):
        if solve_type == "full":
            params = self.params()
        elif solve_type == "pose":
            params = self.pose_params

        if losses_with_weights is None:
            losses_with_weights = dict(
                kpts_losses=1,
                # angle_losses=1,
                edge_losses=100,
                face_losses=100,
            )

        jacobian = np.zeros([jnts_target.size, params.shape[0]])

        pcls = Pointclouds([torch.tensor(pcls_target, dtype=torch.float32, device=device)])

        # accelerate draw
        pcls_vis = pcls_target[np.random.choice(np.arange(pcls_target.shape[0]), size=min(1000, pcls_target.shape[0]), replace=False)]

        losses = LossManager(losses_with_weights, mse_threshold, loss_threshold, plot_type=self.plot_type)

        for i in range(int(max_iter)):
            # update modle
            self.vpose_mapper()
            mesh_updated, jnts_updated = self.model.run(self.params())

            # compute keypoints loss
            loss_kpts, residual = jnts_distance(jnts_updated, jnts_target, activate_distance=kpts_threshold)
            losses.update_loss("kpts_losses", loss_kpts)

            # compute keypoints angle loss
            # loss_angle = angle_loss(jnts_updated, self.pose_params)
            # losses.update_loss("angle_losses", loss_angle)

            # compute edge loss
            if losses_with_weights["edge_losses"] > 0:
                loss_edge = point_mesh_edge_distance(mesh_updated, pcls).cpu()
                losses.update_loss("edge_losses", loss_edge)
        
            # compute face loss
            if losses_with_weights["face_losses"] > 0:
                loss_face = point_mesh_face_distance(mesh_updated, pcls).cpu()
                losses.update_loss("face_losses", loss_face)

            # check loss
            if losses.check_losses():
                break

            for k in range(params.shape[0]):
                jacobian[:, k] = np.hstack([self.get_derivative(k)])
            jtj = np.matmul(jacobian.T, jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            delta = np.matmul(
                np.matmul(np.linalg.inv(jtj), jacobian.T), residual
            ).ravel()
            params -= delta

            update = losses.delta(absolute=False)

            if update > 0 and update > losses.delta(idx=-2, absolute=False):
                u /= v
            else:
                u *= v

            if dbg_level > 0 and i%dbg_level == 0:
                if mesh_updated is None:
                    mesh_updated = [self.model.core.verts - self.pose_params[:self.model.n_coord], self.model.core.faces]
                    
                losses.show_losses()
                if self.plot_type == "matplotlib":
                    vertices, faces= mesh_updated.verts_packed().cpu(), mesh_updated.faces_packed().cpu()
                    losses.show_output_mpl(mesh=(vertices, faces), pcls=pcls_vis, kpts_target=jnts_target, kpts_updated=jnts_updated)
                else:
                    losses.show_output_o3d(mesh_updated, pcls_vis, jnts_target)

            self.update_params(params)
        if dbg_level == 0:
            o3d_plot([o3d_mesh(mesh_updated)])
        return self.params(), losses

    def params(self):
        return np.hstack([self.pose_params, self.shape_params])

    def update_params(self, params: Union[np.lib.npyio.NpzFile, np.ndarray]):
        if isinstance(params, np.lib.npyio.NpzFile):
            self.pose_params = params["pose"]
            self.shape_params = params["shape"]
        elif params.shape[0] == self.model.n_pose + self.model.n_coord:
            self.pose_params = params
        elif params.shape[0] == self.model.n_params - 3 + self.model.n_coord:
            self.pose_params, self.shape_params = params[:self.model.n_pose + self.model.n_coord], params[-self.model.n_shape:]
        else:
            raise RuntimeError("Invalid params")

    def vpose_mapper(self):
        poseSMPL = torch.from_numpy(np.array([self.pose_params[6:69]])).type(torch.float).to('cuda')
        poZ = self.vp.encode(poseSMPL).mean
        self.pose_params[6:69] = self.vp.decode(poZ)['pose_body'].contiguous().reshape(poseSMPL.shape[1]).cpu().numpy()

    def get_derivative(self, n):
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
        params1 = np.array(self.params())
        params2 = np.array(self.params())

        params1[n] += self.eps
        params2[n] -= self.eps

        res1 = self.model.run(params1)[1]
        res2 = self.model.run(params2)[1]

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def save_param(self, file_path):
        np.savez(file_path, pose=self.pose_params, shape=self.shape_params)

    def save_model(self, file_path):
        self.model.core.save_obj(file_path)


def jnts_distance(kpts_updated, kpts_target, activate_distance):
    d = (kpts_updated - kpts_target)
    _filter = np.linalg.norm(d, axis=1) < activate_distance
    residual = np.where(np.repeat(_filter.reshape(_filter.shape[0], 1), 3, axis=1), 0, d).reshape(kpts_updated.size, 1)
    loss_jnts = np.mean(np.square(residual))
    return loss_jnts, residual


def angle_loss(keypoints, pose_params):
    use_feet = keypoints[[19, 20, 21, 22, 23, 24],:].sum() > 0.1
    use_head = keypoints[[15, 16, 17, 18],:].sum() > 0.1
    
    SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14, 20, 21, 22, 23]

    if not use_feet:
        SMPL_JOINT_ZERO_IDX.extend([7, 8])
    if not use_head:
        SMPL_JOINT_ZERO_IDX.extend([12, 15])
    SMPL_POSES_ZERO_IDX = [[j for j in range(3*i, 3*i+3)] for i in SMPL_JOINT_ZERO_IDX]
    SMPL_POSES_ZERO_IDX = np.array(sum(SMPL_POSES_ZERO_IDX, [])) - 3
    # SMPL_POSES_ZERO_IDX.extend([36, 37, 38, 45, 46, 47])
    # return torch.sum(torch.abs(pose_params[SMPL_POSES_ZERO_IDX]))
    return np.sum(np.abs(pose_params[SMPL_POSES_ZERO_IDX]))
