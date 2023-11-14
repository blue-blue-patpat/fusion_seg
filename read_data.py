from PIL import Image
import numpy as np
import torch
from torch import nn 
from pyntcloud import PyntCloud
import open3d
import os
# test1 = Image.open('/remote-home/linxinzhuo/code/lxz_3DSVC/data/label/a1.png')
# test1 = test1.resize((224, 224),resample=Image.BILINEAR)
# png_array = np.array(test1)
# # test2 = np.load('/remote-home/linxinzhuo/code/lxz_3DSVC/data/label/a1.npy')
# # print(test1==png_array)
# a=1

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# a=1

# pc = open3d.io.read_point_cloud('/remote-home/linxinzhuo/code/seq_1/pcd/a1.pcd')
# points = np.asarray(pc.points)
# np.save('/remote-home/linxinzhuo/code/seq_1/pcd/a1.npy', points)
# a=1

pc = np.load('/remote-home/linxinzhuo/code/seq_2/pcd/a1.npy')
a=1

# list_index = [3,4,5,6,7,8,9,10]
# for list_num in list_index:
#     list_path = '/remote-home/linxinzhuo/code/seq_{}/pcd'.format(list_num)
#     for pcd_filename in os.listdir(list_path):
#         pcd_type = pcd_filename.split('.')[1]
#         if pcd_type == 'pcd':
#             pcd_index = pcd_filename.split('.')[0]
#             pcd_filepath = os.path.join(list_path, pcd_filename)

#             pcd = open3d.io.read_point_cloud(pcd_filepath)
#             points = np.asarray(pcd.points)
#             np.save(os.path.join(list_path, pcd_index+'.npy'), points)

