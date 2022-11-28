import os
import numpy as np
import joblib
from mosh.config import SMPLX_MODEL_NEUTRAL_PATH, SMPLX_MODEL_FEMALE_PATH, SMPLX_MODEL_MALE_PATH
from nn.p4t.tools import rodrigues_2_rot_mat
from nn.SMPL.mosh_loss import SMPLXModel
from nn.p4t.tools import copy2cpu as c2c
from visualization.mesh_plot import MoshEvaluateStreamPlot
savepath = '/home/nesc525/drivers/4/VIBE_visual_2'
scene = '2022-03-25_17-10-08_M' 
pred_file = os.path.join(savepath,scene, 'pred.pt')
pred_theta = joblib.load(pred_file)
target_file = os.path.join(savepath,scene, 'target.pt')
target_theta = joblib.load(target_file)
body_model = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0,device = 'cpu')
import torch

def visual(pred_theta,target_theta):
        input_params = dict(
            trans=pred_theta[:,:3],
            pose_body=pred_theta[:,3:-16],
            betas=pred_theta[:,-16:],
            use_rodrigues=True
        )
        pred_output = body_model(**input_params)

        input_params = dict(
            trans=target_theta[:,:3],
            pose_body=target_theta[:,3:-16],
            betas=target_theta[:,-16:],
            use_rodrigues=True
        )
        target_output = body_model(**input_params)
        face_p = pred_output['faces']
        face_g = target_output['faces']
        for frame_id in range(len(pred_output['verts'])):
            if frame_id<=4:
                continue
            mesh_p = pred_output['verts'][frame_id]
            mesh_g = target_output['verts'][frame_id]
            j3d_p = pred_output['joints'][frame_id]
            j3d_g = target_output['joints'][frame_id]
            pred_pelvis = (j3d_p[[2],:] + j3d_p[[3],:]) / 2.0
            target_pelvis = (j3d_g[[2],:] + j3d_g[[3],:]) / 2.0
            mesh_p -= pred_pelvis
            mesh_g -= target_pelvis
            print(frame_id)
            
            yield dict(
                        pred_smpl = dict(
                            mesh = [c2c(mesh_p), c2c(face_p)],
                            color = np.asarray([159, 175, 216]) / 255,
                        ),
                        label_smpl = dict(
                            mesh = [c2c(mesh_g), c2c(face_g)],
                            color = np.asarray([235, 189, 191]) / 255,
                        )
                    )
            shape_err = np.abs(pred_theta[frame_id,-16] - target_theta[frame_id,-16])
            print(shape_err)




gen = visual(torch.tensor(pred_theta),torch.tensor(target_theta))
plot = MoshEvaluateStreamPlot()
plot.show(gen, fps=10, save_path='/home/nesc525/drivers/4/p4t_mosh/RGB/test2/smoke/snapshot/')  