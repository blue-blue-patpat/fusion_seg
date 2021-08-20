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
        self.max_iter = max_iter

        self.params = []

    def solve_full(self, kpts_target, pcls_target, origin_scale, init=None, losses_with_weights=None,
                    mse_threshold=1e-8, loss_threshold=1e-8, u=1e-3, v=1.5, verbose=False):
        if init is None:
            init = np.zeros(self.model.n_params)

        if losses_with_weights is None:
            losses_with_weights = dict(
                kpts_losses=1,
                edge_losses=1,
                face_losses=1,
            )

        jacobian = np.zeros([kpts_target.size, init.shape[0]])

        pcls = Pointclouds([torch.tensor(pcls_target, dtype=torch.float32, device=device)])

        # accelerate draw
        pcls_vis = pcls_target[np.random.choice(np.arange(pcls_target.shape[0]), size=1000, replace=False)]

        losses = LossManager(losses_with_weights, mse_threshold, loss_threshold)

        params = init

        for i in range(self.max_iter):
            # update modle
            mesh_updated, kpts_updated = self.model.run(params)

            # compute keypoints loss
            loss_kpts, residual = keypoints_distance(kpts_updated, kpts_target, activate_distance=40/origin_scale)
            # residual = (kpts_updated - kpts_target).reshape(kpts_updated.size, 1)
            # loss_kpts = np.mean(np.square(residual))
            losses.update_loss("kpts_losses", loss_kpts)

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

            if verbose > 0:
                if i%verbose == 0:
                    vertices, faces= mesh_updated.verts_packed().cpu(), mesh_updated.faces_packed().cpu()
                else:
                    vertices = faces = None
                losses.update_loss_curve(mesh=(vertices, faces), pcls=pcls_vis, kpts_target=kpts_target, kpts_updated=kpts_updated)
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

        pcls = Pointclouds([torch.tensor(pcls_target, dtype=torch.float32, device=device)])

        losses = LossManager(losses_with_weights, mse_threshold, loss_threshold)

        for i in range(self.max_iter):
            # update modle
            mesh_updated, keypoints_updated = self.model.run(np.hstack(self.pose_params, self.shape_params))

            # compute keypoints loss
            residual = (keypoints_updated - kpts_target).reshape(out_n, 1)
            loss_kpts = np.mean(np.square(residual))
            losses.update_loss("kpts_losses", loss_kpts)

            # compute edge loss
            loss_edge = point_mesh_edge_distance(mesh_updated, pcls)
            losses.update_loss("edge_losses", loss_edge)
        
            # compute face loss
            loss_face = point_mesh_face_distance(mesh_updated, pcls)
            losses.update_loss("face_losses", loss_face)

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
                vertices, faces= mesh_updated.verts_packed().cpu(), mesh_updated.faces_packed().cpu()
                losses.update_loss_curve(mesh=(vertices, faces), pcls=pcls_target)
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

def keypoints_distance(kpts_updated, kpts_target, activate_distance):
    d = (kpts_updated - kpts_target)
    _filter = np.linalg.norm(d, axis=1) < activate_distance
    residual = np.where(np.repeat(_filter.reshape(_filter.shape[0], 1), 3, axis=1), 0, d).reshape(kpts_updated.size, 1)
    loss_kpts = np.mean(np.square(residual))
    return loss_kpts, residual
