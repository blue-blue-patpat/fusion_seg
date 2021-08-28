#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   optitrack/calib.py    
@Contact    :   wyzlshx@foxmail.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/26 20:38    wxy        1.0         None
"""

# import lib
import numpy as np
from scipy import optimize
import pandas as pd


def read_csv(path: str) -> np.ndarray:
    """
    Convert OptiTrack CSV file to numpy array
    
    See also set_params().

    :param path: file path
    :return: triangle coordinates
    """
    df = pd.read_csv(path)
    markers_df = df.iloc[6,[i for i, item in enumerate(df.iloc[1]) if "calib" in str(item)]]
    markers_arr = np.asarray(markers_df, np.float64).reshape(-1, 3)
    return markers_arr


def normal_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute normal vector of 3d vector x and vector y.

    :param x: (3,) ndarray
    :param y: (3,) ndarray
    :return: (3,) ndarray
    """
    return np.array([x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]])


def axis_v(coords: np.ndarray) -> np.ndarray:
    """
    Compute unit vector of axis v, according to axis u and axis w(raw).

    :param coords: (4, 3) input coordinates
    :return: (3,) ndarray
    """
    u = axis_u(coords)
    w = _axis_raw_w(coords)
    v = normal_vector(u, w)
    v = v / np.sqrt(np.sum(np.square(u)))
    # check axis v orientation
    if (coords[0] - coords[3]) @ v.T < 0:
        v *= -1
    return v


def axis_u(coords: np.ndarray) -> np.ndarray:
    """
    Compute unit vector of axis u, according to point 0 and point 1.

    :param coords: (4, 3) input coordinates
    :return: (3,) ndarray
    """
    v = coords[0] - coords[1]
    v = v / np.sqrt(np.sum(np.square(v)))
    return v


def _axis_raw_w(coords: np.ndarray) -> np.ndarray:
    """
    Compute raw unit vector of axis w, according to point 0 and point 2.

    May NOT perpendicular to v. axis_w() returns the true value.

    :param coords: (4, 3) input coordinates
    :return: (3,) ndarray
    """
    w = coords[0] - coords[2]
    w = w / np.sqrt(np.sum(np.square(w)))
    return w


def axis_w(coords: np.ndarray) -> np.ndarray:
    """
    Compute unit vector of axis w, according to axis u and axis v.

    :param coords: (4, 3) input coordinates
    :return: (3,) ndarray
    """
    u = axis_u(coords)
    v = axis_v(coords)
    w = normal_vector(u, v)
    w = w / np.sqrt(np.sum(np.square(w)))
    return w


def axis_uvw(coords: np.ndarray) -> dict:
    """
    For visualization use.

    :param coords: (4, 3) input coordinates
    :return:
    """
    return {
        'u': {
            'vector': axis_u(coords),
            'color': 'red',
        },
        'v': {
            'vector': axis_v(coords),
            'color': 'green',
        },
        'w': {
            'vector': axis_w(coords),
            'color': 'blue',
        },
    }


def axis_center(coords: np.ndarray, **kwargs) -> np.ndarray:
    """
    compute uvw axis center
    100 = plate_x/2; 95 = plate_y/2; 50 = plate_radar_distance; 3 = ball_radius

    :param coords: (4, 3) input coordinates
    :param kwargs: d_u, d_v, d_w, r_ball
    :return: center ndarray
    """
    from arbe.config import _d_u, _d_w, _r_ball
    d_u = kwargs.get('d_u', _d_u)
    d_v = kwargs.get('d_v', _r_ball)
    d_w = kwargs.get('d_u', _d_w)
    return coords[0] + d_u/2 * axis_u(coords) - d_w/2 * axis_w(coords) - d_u * axis_u(coords)


def r_x(theta: float) -> np.ndarray:
    """
    Axis x rotate matrix

    :param theta: rotate angle in radian
    :return: rotate matrix (3,3) ndarray
    """
    eqs = np.array([
        [1., 0., 0.],
        [0., np.cos(theta), -np.sin(theta)],
        [0., np.sin(theta), np.cos(theta)]
    ])
    return eqs


def r_y(theta: float) -> np.ndarray:
    """
    Axis y rotate matrix

    :param theta: rotate angle in radian
    :return: rotate matrix (3,3) ndarray
    """
    eqs = np.array([
        [np.cos(theta), 0., np.sin(theta)],
        [0., 1., 0.],
        [-np.sin(theta), 0., np.cos(theta)]
    ])
    return eqs


def r_z(theta: float) -> np.ndarray:
    """
    Axis z rotate matrix

    :param theta: rotate angle in radian
    :return: rotate matrix (3,3) ndarray
    """
    eqs = np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1]
    ])
    return eqs


def r_xyz(**kwargs):
    """
    Return axis x-y-z rotate matrix equations

    :param kwargs: uvw is target axis, list(u,v,w)
    :return: (3,3) ndarray
    """
    u, v, w = kwargs.get('uvw', None)

    def r(angles):
        angle_x, angle_y, angle_z = angles
        R = (r_z(angle_z) @ r_y(angle_y) @ r_x(angle_x))
        eqs = [R[0] @ u.T - 1, R[1] @ v.T - 1, R[2] @ w.T - 1]
        return eqs

    return r


def r_zyx(**kwargs):
    """
    Return axis z-y-x rotate matrix equations

    :param kwargs: uvw is target axis, list(u,v,w)
    :return: (3,3) ndarray
    """
    u, v, w = kwargs.get('uvw', [None, None, None])

    def r(angles):
        angle_x, angle_y, angle_z = angles
        R = (r_x(angle_x) @ r_y(angle_y) @ r_z(angle_z))
        eqs = [R[0] @ u.T - 1, R[1] @ v.T - 1, R[2] @ w.T - 1]
        return eqs

    return r


def set_params(**kwargs) -> tuple:
    """
    Set params d_u, d_v, d_w, r_ball

            ↑z
            ┆
       x←---·
             ↘y

    ·coord3 -d_v
        ↖
    ·coord0---(d_u)---·coord1
       ┆
       ┆
       ┆
     (d_w)       ↘
       ┆          d_v
       ┆          ┌---------┐
       ┆          |  RADAR  |
    ·coord2       └---------┘
    
    :param kwargs: d_u, d_v, d_w, r_ball
    :return: None
    """
    pass
    # _d_u = kwargs.get('d_u', _d_u)
    # _d_v = kwargs.get('d_v', _d_v)
    # _d_w = kwargs.get('d_u', _d_w)
    # _r_ball = kwargs.get('r_ball', _r_ball)
    # return _d_u, _d_v, _d_w, _r_ball


def solve_r(uvw, f=r_zyx, use_angle=False) -> np.ndarray:
    """
    Solve rotate angle.

    :param uvw: list(u,v,w)
    :param f: equations, (3,3) ndarray
    :param use_angle: True for result in angle, False for result in radian, default False
    :return: angle_x, angel_y ,angle_z
    """
    result = optimize.root(f(uvw=uvw), np.array([0, 0, 0]))
    if use_angle:
        return result.x / np.pi * 180
    else:
        return result.x


def solve_t(coords: np.ndarray) -> np.ndarray:
    """
    Solve transfer matrix.

    :param coords: (4, 3) input coordinates
    :return: trans_x, trans_y, trans_z
    """
    return -axis_center(coords)


def trans_matrix(calibrate_coords: np.ndarray) -> tuple:
    """
    Return transform matrix R, t.(z-y-x)

    :param coords: (4, 3) input coordinates
    :return: R: rotate matrix (3,3) ndarray; t: transfer matrix (3,) ndarray
    """
    u = axis_u(calibrate_coords)
    v = axis_v(calibrate_coords)
    w = axis_w(calibrate_coords)
    angles = solve_r([u, v, w], r_zyx)
    angle_x, angle_y, angle_z = angles
    R = r_x(angle_x) @ r_y(angle_y) @ r_z(angle_z)
    t = R @ solve_t(calibrate_coords)
    print("T: ", t, "\nR: ", angles / np.pi * 180)
    return R, t


def trans_coord(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform single coordinate.

    :param coords: (4, 3) input coordinates
    :param R:
    :param t:
    :return: transformed coordinate (3,) ndarray
    """
    return coords @ R.T + t
