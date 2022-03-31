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

def mosh_pkl_parser(pkl_path):
    with open(pkl_path, 'rb') as f:
        mosh_data = pickle.load(f)
    frame_len = len(mosh_data['trans'])
    mosh_result = dict(trans=mosh_data['trans'],
                    betas=mosh_data['betas'][:16],
                    root_orient=mosh_data['fullpose'][:,:3],
                    # root_orient=root_orient,
                    pose_body=mosh_data['fullpose'][:,3:66],
                    pose_hand=mosh_data['fullpose'][:,75:],
                    )
    mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=frame_len, axis=0)
    surface_params = {k: torch.Tensor(v) for k, v in mosh_result.items()}

    body_model = BodyModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH,
                    num_betas=16,
                    num_expressions=0,
                    )
    mesh_result = body_model(**surface_params)

    res = dict(pose = np.hstack((mosh_data['trans'], mosh_data['fullpose'][:,:66])),
                pose_hand = mosh_data['fullpose'][:,75:],
                shape = mosh_data['betas'][:16],
                faces = c2c(body_model.f),
                vertices = c2c(mesh_result.v),
                joints = c2c(mesh_result.Jtr),
                )
    return res


def mosh_single_pkl_parser(param_fname, trans_mat=None):
    mosh_params = np.load(param_fname)

    if trans_mat is not None:
        mosh_offset = mosh_params['joints'][0] - mosh_params['pose'][:3]
        trans = trans_mat['R'] @ (mosh_params['pose'][:3] + mosh_offset) + trans_mat['t'] - mosh_offset
        orient_mat = trans_mat['R'] @ cv2.Rodrigues(mosh_params['pose'][3:6])[0]
        root_orient = cv2.Rodrigues(orient_mat)[0]
    else:
        trans=mosh_params['pose'][:3]
        root_orient=mosh_params['pose'][3:6]
        # full_pose = torch.cat((torch.Tensor(mosh_params['pose'][3:]), torch.Tensor(mosh_params['pose_hand'])), -1)
        # trans_pose = batch_rodrigues(full_pose.view(-1, 3)).view([-1, 3, 3])
        # trans_pose = torch.Tensor(trans_mat['R']) @ trans_pose
        # root_orient = torch.Tensor(trans_mat['R']) @ trans_pose[0]

    mosh_result = dict(trans=trans.reshape(1,-1),
                    root_orient=root_orient.reshape(1,-1),
                    pose_body=mosh_params['pose'][6:].reshape(1,-1),
                    # pose_hand=mosh_params['pose_hand'].reshape(1,-1),
                    betas=mosh_params['shape'].reshape(1,-1),
                    )
    surface_params = {k: torch.Tensor(v) for k, v in mosh_result.items()}

    body_model = BodyModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH,
                    num_betas=16,
                    num_expressions=0,
                    )
    mesh_result = body_model(**surface_params)

    # mesh_verts = rotate_points_xyz(c2c(mesh_result.v), np.array([-90, 0, 0])).reshape(-1, 3)
    # mesh_joints = rotate_points_xyz(c2c(mesh_result.Jtr), np.array([-90, 0, 0])).reshape(-1, 3)

    res = dict(faces = c2c(body_model.f),
                # vertices = mesh_verts,
                # joints = mesh_joints,
                vertices = c2c(mesh_result.v).reshape(-1, 3),
                joints = c2c(mesh_result.Jtr).reshape(-1, 3),
                )
    return res


def mosh_param_parser(mosh_params, body_model=None):
    if body_model is None:
        body_model = BodyModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0)

    batch_size = mosh_params.shape[0]

    mosh_result = dict(trans=mosh_params[:, :3].reshape(batch_size,-1),
                    root_orient=mosh_params[:, 3:6].reshape(batch_size,-1),
                    pose_body=mosh_params[:, 6:-16].reshape(batch_size,-1),
                    betas=mosh_params[:, -16:].reshape(batch_size,-1),
                    )
    surface_params = {k: torch.Tensor(v) for k, v in mosh_result.items()}

    mesh_result = body_model(**surface_params)

    res = dict(faces = c2c(body_model.f),
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