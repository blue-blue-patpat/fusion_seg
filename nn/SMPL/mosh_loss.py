import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import os
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.body_model.lbs import lbs
from mosh.config import SMPLX_MODEL_NEUTRAL_PATH, SMPLX_MODEL_FEMALE_PATH, SMPLX_MODEL_MALE_PATH
from nn.p4t.tools import rodrigues_2_rot_mat

class SMPLXModel(BodyModel):
    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device if device is not None else torch.device('cuda')
        for name in ['init_pose_hand', 'init_pose_jaw','init_pose_eye', 'init_v_template', 'init_expression', 
                    'shapedirs', 'exprdirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights', ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))
        
    def forward(self, trans, pose_body, betas, use_rodrigues=False):
        batch_size = trans.shape[0]
        
        pose_hand = self.init_pose_hand.expand(batch_size, -1)
        pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
        pose_eye = self.init_pose_eye.expand(batch_size, -1)
        v_template = self.init_v_template.expand(batch_size, -1, -1)
        expression = self.init_expression.expand(batch_size, -1)
        

        init_pose = torch.cat([pose_jaw, pose_eye, pose_hand], dim=-1)
        if not use_rodrigues:
            init_pose = rodrigues_2_rot_mat(init_pose)
        full_pose = torch.cat([pose_body, init_pose], dim=-1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                        shapedirs=shapedirs, posedirs=self.posedirs, J_regressor=self.J_regressor,
                        parents=self.kintree_table[0].long(), lbs_weights=self.weights, pose2rot=use_rodrigues)

        joints = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        return dict(verts=verts, joints=joints, faces=self.f)

class MoshLoss(_Loss):
    smplx_model = [None, None, None]
    def __init__(self, device: torch.device = torch.device('cpu'), size_average=None, reduce=None, reduction: str = 'mean', scale: float = 1) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        if self.smplx_model[0] is None:
            self.smplx_model[0] = SMPLXModel(bm_fname=SMPLX_MODEL_FEMALE_PATH, num_betas=16, num_expressions=0, device=device)
        if self.smplx_model[1] is None:
            self.smplx_model[1] = SMPLXModel(bm_fname=SMPLX_MODEL_MALE_PATH, num_betas=16, num_expressions=0, device=device)
        if self.smplx_model[2] is None:
            self.smplx_model[2] = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0, device=device)
        self.scale = scale

    def forward(self, input: torch.Tensor, target: torch.Tensor, use_gender: int = 0, train: bool = True) -> torch.Tensor:

        _input = input * self.scale
        _target = target * self.scale

        if not use_gender:
            input_model = target_model = self.smplx_model[2]
        else:
            input_model = target_model = self.smplx_model[0 if target[0][-1] < 0.5 else 1]

        input_params = dict(
            trans=_input[:, :3],
            pose_body=_input[:, 3:-16],
            betas=_input[:, -16:],
        )

        input_result = input_model(**input_params)
        input_verts = input_result['verts']
        input_joints = input_result['joints']

        target_params = dict(
            trans=_target[:, :3],
            pose_body=_target[:, 3:-16],
            betas=_target[:, -16:],
        )

        target_result = target_model(**target_params)
        target_verts = target_result['verts']
        target_joints = target_result['joints']
        
        per_joint_err = torch.norm((input_joints - target_joints), dim=-1)
        per_vertex_err = torch.norm((input_verts - target_verts), dim=-1)

        if train:
            return (F.l1_loss(input_verts, target_verts, reduction=self.reduction), 
                    F.l1_loss(input_joints, target_joints, reduction=self.reduction))
        else:
            return (torch.sqrt(F.mse_loss(input_verts, target_verts, reduction=self.reduction)),
                    torch.sqrt(F.mse_loss(input_joints, target_joints, reduction=self.reduction)),
                    (per_joint_err, per_vertex_err))
