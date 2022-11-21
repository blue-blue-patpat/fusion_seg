from queue import Queue
import time
import os
import torch
import numpy as np

from nn.p4t.modules.model import P4Transformer
from nn.votenet.utils.pc_util import random_sampling
from visualization.utils import filter_pcl
from nn.SMPL.mosh_loss import SMPLXModel
from mosh.config import SMPLX_MODEL_NEUTRAL_PATH

class ReconstructionDemo():
    def __init__(self, device) -> None:
        checkpoint_path = '/home/nesc525/drivers/4/mm_fusion/mmWave_arbe/pth/checkpoint.pth'
        self.body_model = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0, device=device)
        # Init the model
        self.model = P4Transformer(features=3, radius=0.7, nsamples=32, spatial_stride=32,
                  temporal_kernel_size=3, temporal_stride=1,
                  emb_relu=False,
                  dim=1024, depth=5, heads=8, dim_head=128,
                  mlp_dim=2048, output_dim=151).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        # set model to eval mode (for bn and dp)
        self.model.eval()
        self.device = device
        self.clip = []

    def run_reconstruction(self, pcl, bbox_center, output_dir='', save_data=False):
        pcl[:,:3] -= bbox_center
        pcl = random_sampling(pcl, 1024)
        if not self.clip:
            self.clip =  [pcl for _ in range(5)]
        else:
            self.clip.pop(0)
            self.clip.append(pcl)
        clip = np.asarray(self.clip, dtype=np.float32)
        clip = np.expand_dims(clip.astype(np.float32), 0) # (1,5,1024,6)
    
        # Model inference
        inputs = torch.from_numpy(clip).to(self.device)
        tic = time.time()
        with torch.no_grad():
            output = self.model(inputs)
        output[:,:3] += bbox_center
        pred_mesh = self.body_model(output[:,:3], output[:,3:-16], output[:,-16:])[0]

        toc = time.time()
        print('Inference time: %f'%(toc-tic))
        print('Finished reconstruction.')

        if save_data:
            dump_dir = os.path.join(output_dir, 'recon_results')
            if not os.path.exists(dump_dir): os.mkdir(dump_dir)
            print('Dumped reconstruction results to folder %s'%(dump_dir))

        return pred_mesh