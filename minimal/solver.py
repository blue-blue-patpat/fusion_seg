import os
import numpy as np
import torch
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from alfred.dl.torch.common import device
from minimal.models import KinematicModel, KinematicPCAWrapper
from minimal.utils import LossManager


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
        self.max_iter = int(max_iter)

        self.pose_params = np.zeros(self.model.n_pose)
        self.shape_params = np.zeros(self.model.n_shape)

    def solve(self, jnts_target, pcls_target, solve_type="full", losses_with_weights=None,
                    kpts_threshold=0.04, mse_threshold=1e-8, loss_threshold=1e-8, u=1e-3, v=1.5, verbose=0):
        if solve_type == "full":
            params = self.params()
        elif solve_type == "pose":
            params = self.pose_params

        if losses_with_weights is None:
            losses_with_weights = dict(
                kpts_losses=1,
                angle_losses=1,
                edge_losses=1,
                face_losses=1,
            )

        jacobian = np.zeros([jnts_target.size, params.shape[0]])

        pcls = Pointclouds([torch.tensor(pcls_target, dtype=torch.float32, device=device)])

        # accelerate draw
        pcls_vis = pcls_target[np.random.choice(np.arange(pcls_target.shape[0]), size=1000, replace=False)]

        losses = LossManager(losses_with_weights, mse_threshold, loss_threshold)

        for i in range(self.max_iter):
            # update modle
            mesh_updated, jnts_updated = self.model.run(self.params())

            # compute keypoints loss
            loss_kpts, residual = jnts_distance(jnts_updated, jnts_target, activate_distance=kpts_threshold)
            losses.update_loss("kpts_losses", loss_kpts)

            # compute keypoints angle loss
            loss_angle = angle_loss(jnts_updated, self.pose_params)
            losses.update_loss("angle_losses", loss_angle)

            # compute edge loss
            loss_edge = point_mesh_edge_distance(mesh_updated, pcls).cpu()
            losses.update_loss("edge_losses", loss_edge)
        
            # # compute face loss
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
            print(update)

            if update > 0 and update > losses.delta(idx=-2, absolute=False):
                u /= v
            else:
                u *= v

            if verbose > 0:
                # losses.clear_plt()
                losses.show_losses()

                if i%verbose == 0:
                    vertices, faces= mesh_updated.verts_packed().cpu(), mesh_updated.faces_packed().cpu()
                    losses.show_output(mesh=(vertices, faces), pcls=pcls_vis, kpts_target=jnts_target, kpts_updated=jnts_updated)
                pass
            self.update_params(params)

        return self.params()

    def params(self):
        return np.hstack([self.pose_params, self.shape_params])

    def update_params(self, params: np.ndarray):
        if params.shape[0] == self.model.n_pose:
            self.pose_params = params
        elif params.shape[0] == self.model.n_params - 3:
            self.pose_params, self.shape_params = params[:self.model.n_pose], params[-self.model.n_shape:]
        else:
            raise RuntimeError("Invalid params")

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
