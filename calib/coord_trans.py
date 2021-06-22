#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   coord_trans.py    
@Contact    :   wyzlshx@foxmail.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/10 20:38    wxy        1.0         None
"""

# import lib
import numpy as np
from scipy import optimize
import plotly
import plotly.graph_objs as go


np.set_printoptions(suppress=True)


def read_static_ts(path: str) -> np.ndarray:
    """
    Read NOKOV raw data of triangle markers on radar from .ts file.

    :param path: file path
    :return: triangle coordinates
    """
    with open(path) as f:
        next(f)
        next(f)
        next(f)
        next(f)
        next(f)
        next(f)
        coord_data = f.readline()
        coords = [item for item in coord_data.split('\t') if len(item) > 0][2:-1]
        coord1 = np.array([float(item) for item in coords[0:3]])
        coord2 = np.array([float(item) for item in coords[3:6]])
        coord3 = np.array([float(item) for item in coords[6:9]])
    return np.array([coord1, coord2, coord3])


def normal_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute normal vector of 3d vector x and vector y.

    :param x: (3,) ndarray
    :param y: (3,) ndarray
    :return: (3,) ndarray
    """
    return np.array([x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]])


def axis_u(coords: list) -> np.ndarray:
    """
    Compute unit vector of axis u, according to axis v and axis w(raw).

    :param coords: list(np.ndarray), 3 points' raw coordinates
    :return: (3,) ndarray
    """
    v = axis_v(coords)
    w = _axis_raw_w(coords)
    u = normal_vector(v, w)
    u = u / np.sqrt(np.sum(np.square(u)))
    return u


def axis_v(coords: list) -> np.ndarray:
    """
    Compute unit vector of axis v, according to point 0 and point 1.

    :param coords: list(np.ndarray), 3 points' raw coordinates
    :return: (3,) ndarray
    """
    v = coords[1] - coords[0]
    v = v / np.sqrt(np.sum(np.square(v)))
    return v


def _axis_raw_w(coords: list) -> np.ndarray:
    """
    Compute raw unit vector of axis w, according to point 0 and point 2.

    May NOT perpendicular to v. axis_w() returns the true value.

    :param coords: list(np.ndarray), 3 points' raw coordinates
    :return: (3,) ndarray
    """
    w = coords[0] - coords[2]
    w = w / np.sqrt(np.sum(np.square(w)))
    return w


def axis_w(coords: list) -> np.ndarray:
    """
    Compute unit vector of axis w, according to axis u and axis v.

    :param coords: list(np.ndarray), 3 points' raw coordinates
    :return: (3,) ndarray
    """
    u = axis_u(coords)
    v = axis_v(coords)
    w = normal_vector(u, v)
    w = w / np.sqrt(np.sum(np.square(w)))
    return w


def axis_uvw(coords: list) -> dict:
    """
    For visualization use.

    :param coords: list(np.ndarray), 3 points' raw coordinates
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


def axis_center(coords: list) -> np.ndarray:
    """
    compute uvw axis center
    100 = plate_x/2; 95 = plate_y/2; 50 = plate_radar_distance; 3 = ball_radius

    :param coords: list(np.ndarray), 3 points' raw coordinates
    :return: center ndarray
    """
    return coords[0] + 100 * axis_v(coords) - 95 * axis_w(coords) + (50 - 3) * axis_u(coords)


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


def solve_t(coords: list) -> np.ndarray:
    """
    Solve transfer matrix.

    :param coords: list(np.ndarray), 3 points' raw coordinates
    :return: trans_x, trans_y, trans_z
    """
    return -axis_center(coords)


def trans_matrix(calibrate_coords: list) -> tuple:
    """
    Return transform matrix R, t.(z-y-x)

    :param calibrate_coords: list(np.ndarray), 3 points' raw coordinates
    :return: R: rotate matrix (3,3) ndarray; t: transfer matrix (3,) ndarray
    """
    u = axis_u(calibrate_coords)
    v = axis_v(calibrate_coords)
    w = axis_w(calibrate_coords)
    angles = solve_r([u, v, w], r_zyx)
    angle_x, angle_y, angle_z = angles
    R = r_x(angle_x) @ r_y(angle_y) @ r_z(angle_z)
    t = solve_t(calibrate_coords)
    print("T: ", R @ t, "\nR: ", angles / np.pi * 180)
    return R, t


def trans_coord(coord: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform single coordinate.

    :param coord: list(np.ndarray), 3 points' raw coordinates
    :param R:
    :param t:
    :return: transformed coordinate (3,) ndarray
    """
    return R @ (coord + t)


def visualize_coords(coords=None) -> plotly.graph_objs.Figure:
    """
    Visualize coordinates.

    For notebook, use 'plotly.offline.init_notebook_mode()'

    :param coords: list(np.ndarray), 3 points' raw coordinates
    :return: plotly Figure object
    """
    if coords is None:
        coords = []
    traces = []
    uvw = axis_uvw(coords)
    traces.append(go.Scatter3d(
        x=[item['vector'][0] for item in uvw.values()],
        y=[item['vector'][1] for item in uvw.values()],
        z=[item['vector'][2] for item in uvw.values()],
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
            'color': [item['color'] for item in uvw.values()],
            'colorscale': 'Viridis',  # choose a colorscale
        },
        name='uvw'
    ))
    traces.append(go.Scatter3d(
        x=[item[0] for item in coords / 200],
        y=[item[1] for item in coords / 200],
        z=[item[2] for item in coords / 200],
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
            'color': 'black',
        },
        name='localize_points'
    ))

    for idx, item in uvw.items():
        traces.append(go.Scatter3d(
            x=[0, item['vector'][0]],  # <-- Put your data instead
            y=[0, item['vector'][1]],  # <-- Put your data instead
            z=[0, item['vector'][2]],  # <-- Put your data instead
            mode='lines',
            line=dict(
                color=item['color'],
                width=2
            ),
            name=idx
        ))

    # traces.append(go.Surface(x=[item['vector'][0] for item in uvw.values()],
    #     y=[item['vector'][1] for item in uvw.values()],
    #     z=[item['vector'][2] for item in uvw.values()],))

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    plot_figure = go.Figure(layout=layout, data=traces)

    plot_figure.update_layout(scene=dict(
        xaxis=dict(
            range=[-10, 10],
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white", ),
        yaxis=dict(
            range=[-10, 10],
            backgroundcolor="rgb(230, 200,230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white"),
        zaxis=dict(
            range=[-10, 10],
            backgroundcolor="rgb(230, 230,200)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white", ), ),
        # width=700,
        margin=dict(
            r=10, l=10,
            b=10, t=10)
    )

    # Render the plot.
    return plotly.offline.iplot(plot_figure)
