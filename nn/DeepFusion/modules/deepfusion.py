import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

from nn.DeepFusion.modules.transformer import Transformer as FusionTransformer
from nn.DeepFusion.modules.pointnet2.pointnet2_modules import PointnetSAModule
from nn.DeepFusion.utils import gen_random_indices


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
        pcl_feat = pcls[:,:,3:].permute(0, 2, 1).contiguous() if feat_dim > 3 else None
        xyz, point_feat = self.pointnet2(xyz, pcl_feat)
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

class DeepFusion2(nn.Module):
    def __init__(self, npoint, radius, nsample, dim, depth, heads, dim_head, mlp_dim, output_dim):
        super().__init__()
        self.radar_backbone = PointnetSAModule(npoint=npoint, radius=radius, nsample=nsample, mlp=[3,128,128,dim], use_xyz=True)
        modules = list(resnet18(pretrained=True).children())[:-2]
        modules.append(nn.Conv2d(512, dim, (1, 1)))
        self.image_backbone = torch.nn.Sequential(*modules)
        self.depth_backbone = PointnetSAModule(npoint=npoint, radius=radius, nsample=nsample, mlp=[3,128,128,dim], use_xyz=True)
        self.transformer = FusionTransformer(dim, depth, heads, dim_head, mlp_dim)
        self.radar_feat_num_dim = torch.nn.Linear(49, 1)
        self.image_feat_dim = torch.nn.Linear(1024, 2051)
        self.depth_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_embedding = torch.nn.Linear(3, 2051)
        self.local_embedding = torch.nn.Embedding(5, 2051)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )
        
    def process_image(self, images, pos_emb):
        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.image_backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.image_feat_dim(grid_feat)
        # add embeddings
        embeddings = torch.full((images.shape[0], 1), pos_emb, dtype=torch.long, device=images.device)
        global_pos_emb = self.global_embedding(embeddings)
        local_pos_emb = self.local_embedding(embeddings)
        image_feat = image_feat.view(images.shape[0],1,2048) + global_pos_emb
        grid_feat += local_pos_emb
        
        return grid_feat

    def process_point(self, points, pos_emb):
        # extract cluster features and global point features using a PointNet backbone
        if points.shape[1] == 1024:
            xyz, cluster_feat = self.radar_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
            point_feat = self.radar_feat_num_dim(cluster_feat).squeeze()        
        else:
            xyz, cluster_feat = self.depth_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
            point_feat = self.depth_feat_num_dim(cluster_feat).squeeze()
        cluster_feat = torch.cat([xyz, cluster_feat.transpose(1,2).contiguous()], dim=2)
        # add embeddings
        xyz_embedding = self.point_embedding(xyz)
        cluster_feat += xyz_embedding
        embeddings = torch.full((points.shape[0], 1), pos_emb, dtype=torch.long, device=points.device)
        global_pos_emb = self.global_embedding(embeddings)
        local_pos_emb = self.local_embedding(embeddings)
        point_feat = point_feat.view(points.shape[0],1,2048) + global_pos_emb
        cluster_feat += local_pos_emb
        
        return cluster_feat
    
    def forward(self, args, data_dict, is_train=False):
        batch_size = data_dict['joints_3d'].shape[0]
        if is_train:
            inputs = args.inputs.copy()
            # increase the probability of single input
            input_mask_indices = gen_random_indices(max_random_num=len(inputs))
            # input mask
            for i in input_mask_indices:
                data_dict[inputs[i]] = torch.Tensor([[]])
                inputs[i] = ''
            # conduct image mask if image and point modalities are both active
            if {'master_image','sub_image'} & set(inputs) and {'radar','master_depth','sub_depth'} & set(inputs):
                image_mask = np.ones((batch_size, 1, 1, 1))
                # generate mask indices of batch
                image_indices = gen_random_indices(batch_size, random_ratio=args.image_mask_ratio)
                image_mask[image_indices] = 0.0
                image_mask = torch.from_numpy(image_mask).float().to(args.device)
                image_mask = image_mask.expand(-1, 3, 224, 224)
                if 'master_image' in inputs:
                    data_dict['master_image'] *= image_mask
                if 'sub_image' in inputs:
                    data_dict['sub_image'] *= image_mask
        
        local_feats = []
        
        # extract global and local features
        for i, input in enumerate(['radar','master_image','master_depth','sub_image','sub_depth']):
            if data_dict[input].shape[1]:
                if i == 1 or i == 3:
                    local_feat = self.process_image(data_dict[input], pos_emb=i)
                else:
                    local_feat = self.process_point(data_dict[input], pos_emb=i)
                local_feats.append(local_feat)
                
        local_feats = torch.cat(local_feats, dim=1)
        transformer_output = self.transformer(local_feats)
        output = torch.max(input=transformer_output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        
        return output