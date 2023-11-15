import warnings
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
import numpy as np
from nn.fusion_model.modules.transformer import Transformer as FusionTransformer
from nn.fusion_model.modules.pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
from nn.datasets.utils import gen_random_indices


class DeepFusionSeg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.pc_backbone = PointnetSAModule(
        #     npoint=args.npoint,
        #     radius=args.radius,
        #     nsample=args.nsample,
        #     mlp=[0, 128, 128, args.dim],
        #     use_xyz=True,
        # )
        self.sa_module1 = PointnetSAModule(npoint=256, radius=0.3, nsample=32, mlp=[0, 32, 32, 128], use_xyz=True)
        self.sa_module2 = PointnetSAModule(npoint=128, radius=0.5, nsample=32, mlp=[128, 128, 128, 256], use_xyz=True)
        self.sa_module3 = PointnetSAModule(npoint=64, radius=0.7, nsample=32, mlp=[256, 256, 256, 1024], use_xyz=True)

        self.fp_module3 = PointnetFPModule(mlp=[1024+256, 256, 256])
        self.fp_module2 = PointnetFPModule(mlp=[256+128, 256, 128])
        self.fp_module1 = PointnetFPModule(mlp=[128+0, 128, 128])

        self.point_out = nn.Conv1d(in_channels=128, out_channels=6, kernel_size=1, stride=1, padding=0)

        modules = list(resnet50(pretrained=True).children())[:-4]
        modules.append(nn.Conv2d(512, args.dim, (1, 1)))
        self.image_backbone = torch.nn.Sequential(*modules)
        
        self.transformer = FusionTransformer(args.dim, args.depth, args.heads, args.dim_head, args.mlp_dim)
        self.point_embedding = torch.nn.Linear(3, args.dim)
        self.local_embedding = torch.nn.Embedding(5, args.dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(args.dim),
            nn.Linear(args.dim, args.mlp_dim),
            nn.GELU(),
            nn.Linear(args.mlp_dim, args.output_dim),
        )
        self.f_h, self.f_w = [28, 28]

    def process_image(self, images, pos_emb):
        # extract grid features and global image features using a CNN backbone
        grid_feat = self.image_backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1, 2)
        # add embeddings
        embeddings = torch.full((images.shape[0], 1), pos_emb, dtype=torch.long, device=images.device)
        local_pos_emb = self.local_embedding(embeddings)
        grid_feat += local_pos_emb
        return grid_feat

    # def process_point(self, points, pos_emb):
    #     xyz, cluster_feat = self.pc_backbone(xyz=points[:, :, :3].contiguous())
    #     # add embeddings
    #     xyz_embedding = self.point_embedding(xyz)
    #     cluster_feat = cluster_feat.transpose(1, 2).contiguous() + xyz_embedding
    #     embeddings = torch.full((points.shape[0], 1), pos_emb, dtype=torch.long, device=points.device)
    #     local_pos_emb = self.local_embedding(embeddings)
    #     cluster_feat += local_pos_emb

    #     return cluster_feat

    def resize(self, input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=True):
        if warning:
            if size is not None and align_corners:
                input_h, input_w = tuple(int(x) for x in input.shape[2:])
                output_h, output_w = tuple(int(x) for x in size)
                if output_h > input_h or output_w > output_h:
                    if (
                        (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                        and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)
                    ):
                        warnings.warn(
                            f"When align_corners={align_corners}, "
                            "the output would more aligned if "
                            f"input size {(input_h, input_w)} is `x+1` and "
                            f"out size {(output_h, output_w)} is `nx+1`"
                        )
        return nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    def forward(self, data_dict):
        
        batch_size = data_dict["lidar"].shape[0]
        local_feats = []

        # extract global and local features
        img_local_feat = self.process_image(data_dict['image'], pos_emb=0)
        local_feats.append(img_local_feat)

        l0_xyz, l0_points = data_dict['lidar'], None
        l1_xyz, l1_points = self.sa_module1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa_module2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa_module3(l2_xyz, l2_points)
        point_local_feat = l3_points.transpose(1,2).contiguous()
        pos_feature = torch.full((data_dict['lidar'].shape[0], 1), 1, dtype=torch.long, device=data_dict['lidar'].device)
        point_local_pos_emb = self.local_embedding(pos_feature)
        point_local_feat += point_local_pos_emb
        local_feats.append(point_local_feat)

        # for input in self.args.inputs:
        #     if data_dict[input].shape[1]:
        #         if input in self.args.input_dict["image"]:
        #             local_feat = self.process_image(data_dict[input], pos_emb=0)
        #         else:
        #             # l0_xyz, l0_points = data_dict[input], None
        #             # l1_xyz, l1_points = self.sa_module1(l0_xyz, l0_points)
        #             # l2_xyz, l2_points = self.sa_module2(l1_xyz, l1_points)
        #             # l3_xyz, l3_points = self.sa_module3(l2_xyz, l2_points)
        #             # point_feature = l3_points.transpose(1,2).contiguous()
        #             # pos_feature = torch.full((data_dict[input].shape[0], 1), 1, dtype=torch.long, device=data_dict[input].device)
        #             # local_pos_emb = self.local_embedding(pos_feature)
        #             # point_feature += local_pos_emb
        #             local_feat = self.process_point(data_dict[input], pos_emb=1)
        #         local_feats.append(local_feat)

        local_feats = torch.cat(local_feats, dim=1)
        transformer_output = self.transformer(local_feats)

        ##### 图像分割
        img_seg_logit = self.mlp_head(transformer_output[:, :-64])  ####根据image取前部分，point是后64维特征
        img_seg_logit = img_seg_logit.reshape(batch_size, -1, self.f_h, self.f_w)
        img_seg_logit = self.resize(input = img_seg_logit, size = data_dict["img_label"].shape[1:], mode = "bilinear", align_corners = False)
        _, img_pred_label = img_seg_logit.topk(1, dim=1)
        img_pred_label = img_pred_label.squeeze(1)

        #### 点云分割
        l2_points = self.fp_module3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp_module2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp_module1(l0_xyz, l1_xyz, l0_points, l1_points)
        point_seg_logit = self.point_out(l0_points)

        return img_seg_logit, img_pred_label, point_seg_logit
