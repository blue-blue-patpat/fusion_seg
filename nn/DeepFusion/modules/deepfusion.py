import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

from nn.DeepFusion.modules.transformer import Transformer as FusionTransformer
from nn.DeepFusion.modules.pointnet2.pointnet2_modules import PointnetSAModule


class DeepFusion(nn.Module):
    def __init__(self, npoint, radius, nsample,                                        # Pointnet
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3,                                       # output
                ):                                      
        super().__init__()

        self.pointnet2 = PointnetSAModule(npoint=npoint, radius=radius, nsample=nsample, mlp=[features,128,128,dim], use_xyz=True)
        self.point_pos_embedding = nn.Linear(3, dim)
        
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        modules.append(nn.Conv2d(512, dim, (1, 1)))
        self.resnet = torch.nn.Sequential(*modules)
        self.img_feat_lenth = 49
        self.img_pos_embedding = nn.Embedding(self.img_feat_lenth, dim)

        self.transformer = FusionTransformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, input, train=False):
        batch_size, _, feat_dim = input['pcl'].shape
        device = input['pcl'].get_device()
        
        pcls = input['pcl']
        images = input['img']
        
        if train:
            image_mask = np.ones((batch_size, 1, 1, 1))
            point_mask = np.ones((batch_size, 1, 1))
            pb = np.random.random_sample(2)
            masked_num = np.floor(pb*0.3*batch_size, ) # at most x% of the vertices could be masked
            image_indices = np.random.choice(np.arange(batch_size),replace=False,size=int(masked_num[0]))
            point_indices = list(np.random.choice(np.arange(batch_size),replace=False,size=int(masked_num[1])))
            for idx in image_indices:
                if idx in point_indices:
                    point_indices.remove(idx)
            image_mask[image_indices] = 0.0
            point_mask[point_indices] = 0.0
            image_mask = torch.from_numpy(image_mask).float().to(images.device)
            point_mask = torch.from_numpy(point_mask).float().to(images.device) 
            image_mask = image_mask.expand(-1, 3, 224, 224)
            point_mask = point_mask.expand(-1, 1024, 6)
            images = image_mask * images
            pcls = point_mask * pcls
            
        xyz = pcls[:,:,:3].contiguous()
        point_feat = pcls[:,:,3:].permute(0, 2, 1).contiguous() if feat_dim > 3 else None
        xyz, point_feat = self.pointnet2(xyz, point_feat)
        xyz_embedding = self.point_pos_embedding(xyz)
        emb_point_feat = point_feat.transpose(1,2) + xyz_embedding

        img_feat = self.resnet(images).permute(0, 2, 3, 1).reshape(batch_size, self.img_feat_lenth, -1)

        input_ids = torch.zeros([batch_size, self.img_feat_lenth],dtype=torch.long,device=device)
        position_ids = torch.arange(self.img_feat_lenth, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        img_embedding = self.img_pos_embedding(position_ids)
        emb_img_feat = img_embedding + img_feat

        transformer_output = self.transformer(emb_point_feat, emb_img_feat)
        # transformer_output = self.transformer(emb_point_feat)

        output = torch.cat([point_feat.transpose(1,2), transformer_output], dim=1)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        
        output = self.mlp_head(output)

        return output

