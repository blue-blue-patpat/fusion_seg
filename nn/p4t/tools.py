import numpy as np
import torch
import torch.nn as nn

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).double().to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res

def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)

# def rot_mat_2_euler(R):
#     batch = R.size()[0]
#     sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
#     singular= sy<1e-6
#     singular=singular.float()
        
#     x=torch.atan2(R[:,2,1], R[:,2,2])
#     y=torch.atan2(-R[:,2,0], sy)
#     z=torch.atan2(R[:,1,0],R[:,0,0])
    
#     xs=torch.atan2(-R[:,1,2], R[:,1,1])
#     ys=torch.atan2(-R[:,2,0], sy)
#     zs=R[:,1,0]*0
        
#     out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
#     out_euler[:,0]=x*(1-singular)+xs*singular
#     out_euler[:,1]=y*(1-singular)+ys*singular
#     out_euler[:,2]=z*(1-singular)+zs*singular
#     return out_euler

def rot_mat_2_euler(R):
    batch_size = R.size()[0]
    e = R.reshape(batch_size, -1)
    a = e[:,0]
    f = e[:,1]
    g = e[:,2]
    h = e[:,3]
    k = e[:,4]
    l = e[:,5]
    m = e[:,6]
    n = e[:,7]
    e = e[:,8]
    y = torch.arcsin(torch.clip(g, -1, 1))
    singular= (torch.abs(g)<0.99999).float()
    x = torch.atan2( - l, e)
    z = torch.atan2( - f, a)
    xs = torch.atan2(n, k)
    zs = torch.zeros_like(z)

    out_euler=torch.zeros(batch_size,3).cuda()
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y
    out_euler[:,2]=z*(1-singular)+zs*singular
    return out_euler
    
def euler_2_rot_mat(euler):

    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
    return matrix

# def rotation6d_2_euler(nn_output):
#     batch_size = nn_output.shape[0]
#     num_joints = 9
#     blank_atom=torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32, requires_grad=False, device=torch.device('cuda:0'))
#     q_blank=blank_atom.repeat(batch_size, 1, 1, 1)
#     pose = nn_output[:,3:num_joints*6+3].reshape(batch_size*num_joints, 6).contiguous()
#     tmp_x = nn.functional.normalize(pose[:,:3], dim = -1)
#     tmp_z = nn.functional.normalize(torch.cross(tmp_x, pose[:,3:], dim = -1), dim = -1)
#     tmp_y = torch.cross(tmp_z, tmp_x, dim = -1)

#     tmp_x = tmp_x.view(batch_size,num_joints, 3, 1)
#     tmp_y = tmp_y.view(batch_size,num_joints, 3, 1)
#     tmp_z = tmp_z.view(batch_size,num_joints, 3, 1)
#     pose = torch.cat((tmp_x, tmp_y, tmp_z), -1)
#     R=torch.cat((q_blank,
#                 pose[:,1:3,:,:],
#                 q_blank,
#                 pose[:,3:5,:,:],
#                 q_blank.repeat(1,10,1,1),
#                 pose[:,5:9,:,:],
#                 q_blank.repeat(1,4,1,1)), 1).view(batch_size*24,3,3)
#     rotmat=pose[:,0,:,:]

#     euler = rot_mat_2_euler(R)
#     mat = euler_2_rot_mat(euler)
#     print(mat == R)
#     euler = euler.view(batch_size, -1)

#     return torch.cat((nn_output[:,:3], euler, nn_output[:,-10:]), dim=-1)

def geodesic_loss(mat1, mat2):
    batch_size = mat1.shape[0]
    m1 = mat1.reshape(-1, 3, 3)
    m2 = mat2.reshape(-1, 3, 3)
    total_size = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(total_size).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(total_size).cuda())*-1 )
    
    theta = torch.acos(cos)
    #theta = torch.min(theta, 2*np.pi - theta)
    error = theta.mean()
    return error

def rotation6d_2_rot_mat(rotation6d):
    batch_size = rotation6d.shape[0]
    pose6d = rotation6d.reshape(-1, 6)
    tmp_x = nn.functional.normalize(pose6d[:,:3], dim = -1)
    tmp_z = nn.functional.normalize(torch.cross(tmp_x, pose6d[:,3:], dim = -1), dim = -1)
    tmp_y = torch.cross(tmp_z, tmp_x, dim = -1)

    tmp_x = tmp_x.view(-1, 3, 1)
    tmp_y = tmp_y.view(-1, 3, 1)
    tmp_z = tmp_z.view(-1, 3, 1)
    R = torch.cat((tmp_x, tmp_y, tmp_z), -1)

    return R.reshape(batch_size, -1)

def rodrigues_2_rot_mat(rvecs):
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    total_size = r_vecs.shape[0]
    thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    u = r_vecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.autograd.Variable(torch.zeros([total_size], device="cuda"))  # for broadcasting
    Ks_1 = torch.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = torch.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = torch.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    identity_mat = torch.autograd.Variable(torch.eye(3, device="cuda").repeat(total_size,1,1))
    Rs = identity_mat + torch.sin(thetas).unsqueeze(-1) * Ks + \
         (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    # Avoid returning NaNs where division by zero happened
    R = torch.where(is_zero[:,None,None], identity_mat, Rs)

    return R.reshape(batch_size, -1)

def rot_mat_2_rodrigues(R):
    batch_size = R.shape[0]
    Rs = R.reshape(-1, 3, 3)
    thetas = torch.acos(((Rs[:, 0, 0] + Rs[:, 1, 1] + Rs[:, 2, 2]) - 1) / 2).squeeze()

    is_zero = torch.eq(thetas, torch.tensor(0.0))
    
    multi = torch.where(is_zero, torch.autograd.Variable(torch.zeros_like(thetas)), 1 / (2 * torch.sin(thetas))) ##
    
    rx = multi * (Rs[:, 2, 1] - Rs[:, 1, 2]) * thetas
    ry = multi * (Rs[:, 0, 2] - Rs[:, 2, 0]) * thetas
    rz = multi * (Rs[:, 1, 0] - Rs[:, 0, 1]) * thetas

    rvecs = torch.stack((rx, ry, rz), -1)
    return rvecs.reshape(batch_size, -1)

def rotation6d_2_rodrigues(rotation6d):
    batch_size = rotation6d.shape[0]
    R = rotation6d_2_rot_mat(rotation6d)
    rvecs = rot_mat_2_rodrigues(R).view(batch_size, -1)
    
    return rvecs
    
def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()
