import pickle
import os
import torch
import cv2
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.rotation_tools import rotate_points_xyz
from nn.p4t.tools import rodrigues_2_rot_mat, rot_mat_2_rodrigues

def mosh_pkl_parser(pkl_path):
    with open(pkl_path, 'rb') as f:
        mosh_data = pickle.load(f)
    time_length = len(mosh_data['trans'])
    mosh_result = dict(trans=mosh_data['trans'],
                    betas=mosh_data['betas'][:16],
                    root_orient=mosh_data['fullpose'][:,:3],
                    # root_orient=root_orient,
                    pose_body=mosh_data['fullpose'][:,3:66],
                    pose_hand=mosh_data['fullpose'][:,75:])
    mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=time_length, axis=0)
    surface_parms = {k: torch.Tensor(v) for k, v in mosh_result.items()}

    sm = BodyModel(bm_fname='/home/nesc525/chen/mosh/soma/support_files/smplx/neutral/model.npz',
                    num_betas=16,
                    num_expressions=0,
                    num_dmpls=None,
                    dmpl_fname=None)
    mesh_result = sm(**surface_parms)

    # mesh_verts = rotate_points_xyz(c2c(mesh_result.v), np.array([-90, 0, 0]))
    # mesh_joints = rotate_points_xyz(c2c(mesh_result.Jtr), np.array([-90, 0, 0]))

    res = dict(pose = np.hstack((mosh_data['trans'], mosh_data['fullpose'][:,:66])),
                pose_hand = mosh_data['fullpose'][:,75:],
                shape = mosh_data['betas'][:16],
                faces = c2c(sm.f),
                # vertices = mesh_verts,
                # joints = mesh_joints,
                vertices = c2c(mesh_result.v),
                joints = c2c(mesh_result.Jtr),
                )
    return res


def mosh_param_parser(param_fname):
    mosh_params = np.load(param_fname)
    mosh_result = dict(trans=mosh_params['pose'][:3].reshape(1,-1),
                    betas=mosh_params['shape'].reshape(1,-1),
                    root_orient=mosh_params['pose'][3:6].reshape(1,-1),
                    pose_body=mosh_params['pose'][6:].reshape(1,-1),
                    pose_hand=mosh_params['pose_hand'].reshape(1,-1))
    surface_parms = {k: torch.Tensor(v) for k, v in mosh_result.items()}

    sm = BodyModel(bm_fname='/home/nesc525/chen/mosh/soma/support_files/smplx/neutral/model.npz',
                    num_betas=16,
                    num_expressions=0,
                    num_dmpls=None,
                    dmpl_fname=None)
    mesh_result = sm(**surface_parms)

    # mesh_verts = rotate_points_xyz(c2c(mesh_result.v), np.array([-90, 0, 0])).reshape(-1, 3)
    # mesh_joints = rotate_points_xyz(c2c(mesh_result.Jtr), np.array([-90, 0, 0])).reshape(-1, 3)

    res = dict(faces = c2c(sm.f),
                # vertices = mesh_verts,
                # joints = mesh_joints,
                vertices = c2c(mesh_result.v).reshape(-1, 3),
                joints = c2c(mesh_result.Jtr).reshape(-1, 3),
                )
    return res


def get_spec_files(path, suffix):
    file_list = []
    if os.path.exists(path):
        f_list = os.listdir(path)
        for f in f_list:
            if os.path.splitext(f)[1] == suffix:
                file_list.append(f)
    return file_list