import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18
from nn.fusion_model.modules.pointnet2.pointnet2_modules import PointnetSAModule

from nn.fusion_model.modules.transformer import Transformer2 as FusionTransformer
from nn.fusion_model.utils import FFN, PredictorLG
from nn.datasets.utils import gen_random_indices, project_pcl_torch

class TokenFusion(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.radar_backbone = PointnetSAModule(npoint=args.npoint, radius=args.radius, nsample=args.nsample, 
                                               mlp=[args.features,128,128,args.dim], use_xyz=True)
        modules = list(resnet18(pretrained=True).children())[:-2]
        modules.append(nn.Conv2d(512, args.dim, (1, 1)))
        self.image_backbone = nn.Sequential(*modules)
        self.depth_backbone = PointnetSAModule(npoint=args.npoint, radius=args.radius, nsample=args.nsample, 
                                               mlp=[args.features,128,128,args.dim], use_xyz=True)
        self.image_trans_encoder = FusionTransformer(args.dim, args.depth, args.heads, args.dim_head, args.mlp_dim)
        self.point_trans_encoder = copy.deepcopy(self.image_trans_encoder)
        self.score_net = nn.ModuleList([PredictorLG(args.dim) for _ in range(args.depth)])
        self.patch_mlp = nn.ModuleList([FFN(args.dim, args.mlp_dim, args.dim, 3) for _ in range(args.depth)])
        self.point_embedding = nn.Linear(3, args.dim)
        self.local_embedding = nn.Embedding(5, args.dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(args.dim),
            nn.Linear(args.dim, args.mlp_dim),
            nn.GELU(),
            nn.Linear(args.mlp_dim, args.output_dim),
        )
    
    def get_img_patch_idx(self, xyz, trans_mat, cam, bbox):
        proj_uv = project_pcl_torch(xyz, trans_mat, cam=cam)
        bbox_size = bbox[:,2:] - bbox[:,:2]
        trans_uv = (proj_uv - bbox[:,None,:2])*224/bbox_size[:,None]
        img_patch_idx = trans_uv[:,:,0] // 32 * 7 + trans_uv[:,:,1] // 32
        return img_patch_idx.long().unsqueeze(-1).expand(-1,-1,self.args.dim)
    
    def process_image(self, images, pos_emb):
        # extract grid features and global image features using a CNN backbone
        grid_feat = self.image_backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        # add embeddings
        embeddings = torch.full((images.shape[0], 1), pos_emb, dtype=torch.long, device=images.device)
        local_pos_emb = self.local_embedding(embeddings)
        grid_feat += local_pos_emb
        
        return grid_feat

    def process_point(self, points, pos_emb):
        # extract cluster features and global point features using a PointNet backbone
        if points.shape[1] == 1024:
            xyz, cluster_feat = self.radar_backbone(xyz=points[:,:,:3].contiguous(),
                                                    features=points[:,:,3:].transpose(1,2).contiguous())
        else:
            xyz, cluster_feat = self.depth_backbone(xyz=points[:,:,:3].contiguous(),
                                                    features=points[:,:,3:].transpose(1,2).contiguous())
        # add embeddings
        xyz_embedding = self.point_embedding(xyz)
        cluster_feat = cluster_feat.transpose(1,2).contiguous() + xyz_embedding
        embeddings = torch.full((points.shape[0], 1), pos_emb, dtype=torch.long, device=points.device)
        local_pos_emb = self.local_embedding(embeddings)
        cluster_feat += local_pos_emb
        
        return cluster_feat, xyz
    
    def forward(self, data_dict, is_train=False):
        # image and point mask
        if is_train:
            batch_size = data_dict['radar'].shape[0]
            image_mask = np.ones((batch_size, 1, 1, 1))
            # generate mask indices of batch
            image_indices = gen_random_indices(batch_size, random_ratio=0.3)
            image_mask[image_indices] = 0.0
            image_mask = torch.from_numpy(image_mask).float().to(self.args.device)
            image_mask = image_mask.expand(-1, 3, 224, 224)
            data_dict['master_image'] *= image_mask
            data_dict['sub_image'] *= image_mask
        
        image_feats = []
        point_feats = []
        proj_patch_idx = []
        # extract global and local features
        for i, input in enumerate(['radar','master_image','master_depth','sub_image','sub_depth']):
            if data_dict[input].shape[1]:
                if i == 1 or i == 3:
                    image_feats.append(self.process_image(data_dict[input], pos_emb=i))
                else:
                    point_feat, xyz = self.process_point(data_dict[input], pos_emb=i)
                    point_feats.append(point_feat)
                    mas_patch_idx = self.get_img_patch_idx(xyz+data_dict['root_pelvis'][:,None,:], 
                                                           data_dict['trans_mat']['master_image'].cuda(), 
                                                           'master', data_dict['bbox']['master_image'].cuda())
                    sub_patch_idx = self.get_img_patch_idx(xyz+data_dict['root_pelvis'][:,None,:], 
                                                           data_dict['trans_mat']['sub_image'].cuda(), 
                                                           'sub', data_dict['bbox']['sub_image'].cuda())
                    proj_patch_idx.append([mas_patch_idx, sub_patch_idx])

        pred_score = []
        # token fusion
        for i in range(self.args.depth):
            image_tokens = list(map(self.image_trans_encoder.layers[i], image_feats))
            point_tokens = list(map(self.point_trans_encoder.layers[i], point_feats))
            # score the tokens
            token_scores = F.softmax(self.score_net[i](torch.cat(image_tokens+point_tokens, dim=1)), -1)[:,:,0]
            pred_score.append(token_scores)
            token_masks = torch.where(token_scores > 0.02, 1, 0).unsqueeze(-1)
            image_token_masks = torch.split(token_masks[:,:49*2], 49, 1)
            point_token_masks = torch.split(token_masks[:,49*2:], 49, 1)
            # replace unimportant tokens
            for j in range(len(image_tokens)):
                image_feats[j] = image_tokens[j] * image_token_masks[j] + image_tokens[1-j] * ~image_token_masks[j]
            for k in range(len(point_tokens)):
                mas_img_token = torch.gather(image_tokens[0], 1, proj_patch_idx[k][0])
                sub_img_token = torch.gather(image_tokens[1], 1, proj_patch_idx[k][1])
                patch_feat = self.patch_mlp[i](mas_img_token) + self.patch_mlp[i](sub_img_token)
                point_feats[k] = point_tokens[k] * point_token_masks[k] + patch_feat * ~point_token_masks[k]
        
        output = torch.max(torch.cat(image_feats+point_feats, 1), dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        
        pred_score = torch.cat(pred_score, dim=1).squeeze()
        pred_dict = {'pred_score': pred_score, 'smplx_param': output}

        return pred_dict