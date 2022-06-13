import pickle
import os
import torch
import cv2
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.rotation_tools import rotate_points_xyz
from human_body_prior.body_model.lbs import batch_rodrigues
from nn.p4t.tools import rodrigues_2_rot_mat, rot_mat_2_rodrigues
from mosh.config import SMPLX_MODEL_NEUTRAL_PATH

def mosh_pose_transform(trans, root_orient, root_joint, trans_mat):
    mosh_offset = root_joint - trans
    new_trans = trans_mat['R'] @ (trans + mosh_offset) + trans_mat['t'] - mosh_offset
    orient_mat = trans_mat['R'] @ cv2.Rodrigues(root_orient)[0]
    new_orient = cv2.Rodrigues(orient_mat)[0]
    return new_trans, new_orient

def get_mosh_params(fname, type):
    pass

def mesh_from_mosh_params(mosh_params, trans_mat=None, body_model=None):
    if trans_mat is not None:
        trans, root_orient = mosh_pose_transform(mosh_params['pose'][:3], mosh_params['pose'][3:6], mosh_params['joints'][0], trans_mat)
    else:
        trans=mosh_params['pose'][:3]
        root_orient=mosh_params['pose'][3:6]

    pose_body = mosh_params['pose'][6:]
    betas = mosh_params['shape']

    mosh_result = dict(
        trans=trans.reshape(1,-1),
        root_orient=root_orient.reshape(1,-1),
        pose_body=pose_body.reshape(1,-1),
        betas=betas.reshape(1,-1),
    )
    surface_params = {k: torch.Tensor(v) for k, v in mosh_result.items()}
    
    if body_model is None:
        body_model = BodyModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0)
    mesh_result = body_model(**surface_params)

    res = dict(
        faces = c2c(body_model.f),
        vertices = c2c(mesh_result.v).reshape(-1, 3),
        joints = c2c(mesh_result.Jtr).reshape(-1, 3),
    )
    return res


def mesh_from_mosh_batch(mosh_params, body_model=None):
    if body_model is None:
        body_model = BodyModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0)

    batch_size = mosh_params.shape[0]

    mosh_result = dict(
        trans=mosh_params[:, :3].reshape(batch_size,-1),
        root_orient=mosh_params[:, 3:6].reshape(batch_size,-1),
        pose_body=mosh_params[:, 6:-16].reshape(batch_size,-1),
        betas=mosh_params[:, -16:].reshape(batch_size,-1),
    )
    surface_params = {k: torch.Tensor(v) for k, v in mosh_result.items()}

    mesh_result = body_model(**surface_params)

    res = dict(
        faces = c2c(body_model.f),
        vertices = c2c(mesh_result.v),
        joints = c2c(mesh_result.Jtr),
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