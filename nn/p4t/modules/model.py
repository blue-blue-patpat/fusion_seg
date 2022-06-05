import torch
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from transformer import *
from torchvision.models import resnet18


class P4Transformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,3:].permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        return output


class ImageFeatureFusion(P4Transformer):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3):                                      # output

        super().__init__(radius, nsamples, spatial_stride,
                        temporal_kernel_size, temporal_stride,
                        emb_relu,
                        dim, depth, heads, dim_head,
                        mlp_dim, output_dim, features)

        resnet = resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(resnet, torch.nn.Linear(1000, 100))
        self.features = features
        
    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input['pcl'].get_device()
        batch_size, clip_len, num_points, _ = input['pcl'].shape
        rgb_features = self.resnet(input['img'].view(-1, 224, 224, 3).permute(0, 3, 1, 2))
        rgb_features = rgb_features.reshape(batch_size, clip_len, 1, self.features).repeat(1, 1, num_points, 1)
        xyzs, features = self.tube_embedding(input['pcl'][:,:,:,:3], torch.cat((input['pcl'][:,:,:,3:], rgb_features), -1).permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        return output


from kinect.config import INTRINSIC, MAS
from torchvision.transforms import Resize
class FeatureMapFusion(P4Transformer):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3):                                      # output

        super().__init__(radius, nsamples, spatial_stride,
                        temporal_kernel_size, temporal_stride,
                        emb_relu,
                        dim, depth, heads, dim_head,
                        mlp_dim, output_dim, features)

        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        modules.append(torch.nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, padding=1))
        modules.append(torch.nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True))
        self.resnet = torch.nn.Sequential(*modules)
        self.resize_d = Resize([224, 224])
        self.resize_u = Resize([1536, 2048])
        self.features = features

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input['pcl'].get_device()
        batch_size, clip_len, num_points, _ = input['pcl'].shape

        # Project to 2d
        intrinsic = torch.tensor(INTRINSIC[MAS], dtype=torch.float32, device=device)
        pcl = input['pcl']/input['pcl'][:,:,:,2].unsqueeze(-1)
        pcl_2d = (intrinsic.repeat(batch_size, clip_len, num_points, 1, 1) @ pcl.unsqueeze(-1))[:,:,:,:2]
        pcl_2d = pcl_2d.reshape(batch_size*clip_len, num_points, 1, -1)
        pcl_2d[:,:,:,0] = pcl_2d[:,:,:,1]/1536
        pcl_2d[:,:,:,1] = pcl_2d[:,:,:,0]/2048
        
        feature_map = self.resnet(self.resize_d(input['img'].reshape(-1, 1536, 2048, 3).permute(0, 3, 1, 2)))
        feature_map = self.resize_u(feature_map)
        rgb_features = torch.nn.functional.grid_sample(feature_map, pcl_2d, align_corners=False)
        rgb_features = rgb_features.reshape(batch_size, clip_len, 3, num_points).permute(0, 1, 3, 2)
        xyzs, features = self.tube_embedding(input['pcl'][:,:,:,:3], torch.cat((input['pcl'][:,:,:,3:], rgb_features), -1).permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        return output