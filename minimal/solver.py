from minimal.models import KinematicPCAWrapper
from tqdm import tqdm
import numpy as np
from vctoolkit import Timer
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance


class Solver:
  def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8, verbose=False):
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
    self.eps = eps
    self.max_iter = max_iter
    self.loss_threshold = mse_threshold
    self.verbose = verbose
    self.timer = Timer()

  def get_derivative(self, model, params, n):
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

    res1 = model.run(params1)[1]
    res2 = model.run(params2)[1]

    d = (res1 - res2) / (2 * self.eps)

    return d.ravel()

  def solve(self, model: KinematicPCAWrapper, pcls_target, kpts_target, init=None, u=1e-3, v=1.5):
    """
    Solver for the target.

    Parameters
    ----------
    model : object
      Wrapper to be manipulated.
    target : np.ndarray
      Optimization target.
    init : np,ndarray, optional
      Initial parameters, by default None
    u : float, optional
      LM algorithm parameter, by default 1e-3
    v : float, optional
      LM algorithm parameter, by default 1.5

    Returns
    -------
    np.ndarray
      Solved model parameters.
    """
    if init is None:
      init = np.zeros(model.n_params)

    out_n = np.shape(kpts_target.flatten())[0] + 2
    jacobian = np.zeros([out_n, init.shape[0]])

    w_kpts = 0.1
    w_edge = 5
    w_face = 5

    self.kpts_losses = []
    self.edge_losses = []
    self.face_losses = []
    self.losses = []
    self.updates = []

    params = init
    for i in range(self.max_iter):
      # update modle
      mesh_updated, keypoints_updated = model.run(params)
      
      # compute keypoints loss
      residual = (keypoints_updated - kpts_target).reshape(out_n - 2, 1)
      loss_kpts = np.mean(np.square(residual))
      self.kpts_losses.append(loss_kpts)

      # compute edge loss
      loss_edge = point_mesh_edge_distance(mesh_updated, pcls_target)
      self.edge_losses.append(loss_edge)

      # compute face loss
      loss_face = point_mesh_face_distance(mesh_updated, pcls_target)
      self.face_losses.append(loss_face)

      loss = w_kpts * loss_kpts + w_edge * loss_edge + w_face * loss_face

      residual = np.vstack([residual, w_edge * loss_edge, w_face * loss_face])

      # check loss
      if i > 0:
        last_loss = self.losses[-1]
        last_update = self.updates[-1]
      else:
        last_loss = 0
        last_update = 0

      if abs(loss - last_loss) < self.loss_threshold:
        return params

      for k in range(params.shape[0]):
        jacobian[:, k] = np.hstack([self.get_derivative(model, params, k), 0, 0])
      jtj = np.matmul(jacobian.T, jacobian)
      jtj = jtj + u * np.eye(jtj.shape[0])

      update = last_loss - loss
      delta = np.matmul(
        np.matmul(np.linalg.inv(jtj), jacobian.T), residual
      ).ravel()
      params -= delta

      if update > last_update and update > 0:
        u /= v
      else:
        u *= v

      self.updates.append(update)
      self.losses.append(loss)

      if self.verbose:
        print(i, self.timer.tic(), loss)

    return params
