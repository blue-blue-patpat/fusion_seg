import numpy as np
import math


class JointsBridge():
    def __init__(self) -> None:
        self.k_jnts = np.zeros((32, 3))
        self.smpl_jnts = np.zeros((28, 3))

    def smpl_from_kinect(self, jnts, pcl=np.array([[0,0,0]])):
        '''
        self.kinect_joints_transfer_coordinates(jnts, pcl)
        self.filter_pcl()
        '''
        self.update_smpl_joints_from_kinect_joints(jnts)
        self.smpl_joints_transfer_coordinates(self.smpl_jnts,pcl)
        self.filter_pcl()
        return self.normalization()

    def filter_pcl(self, kinect_joints=None, pcl=None) -> np.ndarray:
        if kinect_joints is None:
            kinect_joints = self.k_jnts
        if pcl is None:
            pcl = self.pcl
        upper_bound = kinect_joints.max(axis=0) + 300
        lower_bound = kinect_joints.min(axis=0) - 300
        # remove ground pcl
        lower_bound[1] += 320
        _filter = np.apply_along_axis(
            lambda row:\
                np.all((lower_bound<row) & (row<upper_bound))\
                and np.linalg.norm(kinect_joints - row, axis=1, keepdims=True).min() < 250,\
        axis=1, arr=pcl)

        self.pcl = pcl[_filter]
        return self.pcl

    def kinect_joints_transfer_v2(self, kinect_joints=None):
        from calib.coord_trans import solve_r, r_zyx, r_x, r_y, r_z
        if kinect_joints is None:
            kinect_joints = self.k_jnts
        content=kinect_joints[:,0:3]
        u = np.array([1,0,0])
        v = np.array([0,-1,0])
        w = np.array([0,0,-1])
        angles = solve_r([u, v, w], r_zyx)
        angle_x, angle_y, angle_z = angles
        R = r_x(angle_x) @ r_y(angle_y) @ r_z(angle_z)
        t = -content[0]
        self.k_jnts = (content + np.repeat([t], content.shape[0], axis=0)) @ R.T
        return self.k_jnts
    
    def kinect_joints_transfer_coordinates(self, kinect_joints=None, pcl=None): 
        if kinect_joints is None:
            kinect_joints = self.k_jnts
        if pcl is None:
            pcl = self.pcl
        # extract first 3 rows as joints(x, y, z)
        content=kinect_joints[:,:3]
        pcl = pcl[:,:3]

        # tranfer
        t = -np.array([0.75 * content[2] + 0.25 * content[1]])

        # apply transfer to all joints
        content = content + np.repeat(t, content.shape[0], 0)

        '''
        point_cloud1 = open3d.geometry.PointCloud()
        point_cloud1.points = open3d.utility.Vector3dVector(content)
        point_cloud1.paint_uniform_color([1,0,0])
        '''
        g=(content[4,:]+content[11,:])/2  #旋转r1到y轴

        h=[[1, 0, 0],   #绕x旋转矩阵
            [0, included_cos(g[1],g[2]), -included_cos(g[2],g[1])],
            [0, included_cos(g[2],g[1]), included_cos(g[1],g[2])]]

        i=np.dot(g,h)

        j=[[included_cos(i[1], i[0]), -included_cos(i[0],i[1]),0], #绕z旋转矩阵
            [included_cos(i[0],i[1]), included_cos(i[1],i[0]), 0],
            [0, 0, 1]]

        k=np.dot(h,j)
        l=np.dot(content,k)

        m=[[included_cos(l[4,0],l[4,2]),0,included_cos(l[4,2],l[4,0])],  #绕y旋转矩阵
            [0,1,0],
            [-included_cos(l[4,2],l[4,0]),0,included_cos(l[4,0],l[4,2])]]
        
        self.k_jnts = np.dot(l,m)   #乘到最后是前三列最终
        R = np.dot(k,m)    #matrix是最终旋转矩阵

        self.pcl = np.dot(pcl + np.repeat(t, pcl.shape[0], 0), R)
        return self.k_jnts, self.pcl
        '''
        pt2=open3d.geometry.PointCloud()
        pt2.points=open3d.utility.Vector3dVector(n.reshape(-1, 3))
        pt2.paint_uniform_color([0, 1, 0])

        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
        open3d.visualization.draw_geometries([point_cloud1,pt2]+[axis_pcd])
        '''

    def smpl_joints_transfer_coordinates(self, smpl_joints=None, pcl=None):
        """
        :param smpl_joints: n*3 array
        """
        if smpl_joints is None:
            smpl_joints = self.smpl_jnts
        if pcl is None:
            pcl = self.pcl
        # tranfer
        self.t = -np.array([smpl_joints[6]]) #坐标规范下6号点为原点

        # apply transfer to all joints
        smpl_joints = smpl_joints + np.repeat(self.t, smpl_joints.shape[0], 0)

        g=smpl_joints[9,:]  #旋转r1到y轴

        h=[[1, 0, 0],   #绕x旋转矩阵
            [0, included_cos(g[1],g[2]), -included_cos(g[2],g[1])],
            [0, included_cos(g[2],g[1]), included_cos(g[1],g[2])]]

        i=np.dot(g,h)

        j=[[included_cos(i[1], i[0]), -included_cos(i[0],i[1]),0], #绕z旋转矩阵
            [included_cos(i[0],i[1]), included_cos(i[1],i[0]), 0],
            [0, 0, 1]]

        k=np.dot(h,j)
        l=np.dot(smpl_joints,k)

        m=[[included_cos(l[13,0],l[13,2]),0,included_cos(l[13,2],l[13,0])],  #绕y旋转矩阵
            [0,1,0],
            [-included_cos(l[13,2],l[13,0]),0,included_cos(l[13,0],l[13,2])]]
        
        self.smpl_jnts = np.dot(l,m)   #乘到最后是前三列最终
        self.R = np.dot(k,m)    #matrix是最终旋转矩阵
        self.pcl = np.dot(pcl + np.repeat(self.t, pcl.shape[0], 0), self.R)
        return self.smpl_jnts, self.pcl

    def update_smpl_joints_from_kinect_joints(self, kinect_joints=None):
        if kinect_joints is None:
            kinect_joints = self.k_jnts
        # see also: https://docs.microsoft.com/bs-latn-ba/azure/kinect-dk/body-joints#joint-coordinates
        self.smpl_jnts = np.array([
            # SMPL              Kinect
            # 0_pelvis          0.5*0_PELVIS+0.5*1_SPINE_NAVAL
            0.5 * kinect_joints[0] + 0.5 * kinect_joints[1],
            # 1_left leg root   18_HIP_LEFT
            kinect_joints[18],
            # 2_right leg root  22_HIP_RIGHT
            kinect_joints[22],
            # 3_lowerback       1_SPINE_NAVAL
            kinect_joints[1],
            # 4_left knee       19_KNEE_LEFT
            kinect_joints[19],
            # 5_right knee      23_KNEE_RIGHT
            kinect_joints[23],
            # 6_upperback       0.75*2_SPINE_CHEST+0.25*1_SPINE_NAVAL
            0.65 * kinect_joints[2] + 0.35 * kinect_joints[1],
            # 7_left ankle      20_ANKLE_LEFT
            kinect_joints[20],
            # 8_right ankle     24_ANKLE_RIGHT
            kinect_joints[24],
            # 9_thorax          2_SPINE_CHEST
            kinect_joints[2],
            # 10_left toes      21_FOOT_LEFT
            kinect_joints[21],
            # 11_right toes     25_FOOT_RIGHT
            kinect_joints[25],
            # 12_lowerneck      3_NECK
            kinect_joints[3],
            # 13_left clavicle  0.4*4_CLAVICLE_LEFT+0.3*2_SPINE_CHEST+0.3*5_SHOULDER_LEFT
            0.4*kinect_joints[4]+0.3*kinect_joints[2]+0.3*kinect_joints[5],
            # 14_right clavicle 0.4*11_CLAVICLE_RIGHT+0.3*2_SPINE_CHEST+0.3*12_SHOULDER_RIGHT
            0.4*kinect_joints[11]+0.3*kinect_joints[2]+0.3*kinect_joints[12],
            # 15_upperneck      26_HEAD
            kinect_joints[26],
            # 16_left armroot   0.9*5_SHOULDER_LEFT+0.1*4_CLAVICLE_LEFT
            0.9*kinect_joints[5]+0.1*kinect_joints[4],
            # 17_right armroot  0.9*12_SHOULDER_RIGHT+0.1*11_CLAVICLE_RIGHT
            0.9*kinect_joints[12]+0.1*kinect_joints[11],
            # 18_left elbow     6_ELBOW_LEFT
            kinect_joints[6],
            # 19_right armroot  13_ELBOW_RIGHT
            kinect_joints[13],
            # 20_left wrist     7_WRIST_LEFT
            kinect_joints[7],
            # 21_right wrist    14_WRIST_RIGHT
            kinect_joints[14],
            # 22_left hand      8_HAND_LEFT
            kinect_joints[8],
            # 23_right hand     15_HAND_RIGHT
            kinect_joints[15],
            # ext_left finger tip   9_HANDTIP_LEFT
            kinect_joints[9],
            # ext_right finger tip  16_HANDTIP_RIGHT
            kinect_joints[16],
            # ext_left toe tip      21_FOOT_LEFT+0.3*(21_FOOT_LEFT-20_ANKLE_LEFT)
            kinect_joints[21] + 0.3*(kinect_joints[21]-kinect_joints[20]),
            # ext_right toe tip     25_FOOT_RIGHT+0.3*(25_FOOT_RIGHT-24_ANKLE_RIGHT)
            kinect_joints[25] + 0.3*(kinect_joints[25]-kinect_joints[24]),
            # ext_head_top          27_NOSE+1.5*((28_EYE_LEFT-27_NOSE)+(30_EYE_RIGHT-27_NOSE))
            kinect_joints[3] + 1*((kinect_joints[29]-kinect_joints[3]) + (kinect_joints[31]-kinect_joints[3]))
        ])
        return self.smpl_jnts

    def smpl_from_opti(self,opti_jnts):
        self.smpl_jnts = np.array([
            # SMPL              Kinect
            # 0_pelvis          0.8*0_PELVIS+0.2*1_SPINE_NAVAL
            0.8*(0.5*opti_jnts(4)+0.5*opti_jnts(5))+0.2*0.5*(0.2*opti_jnts(25)+0.4*(opti_jnts(0)+opti_jnts(2))+0.2*opti_jnts(31)+0.4*(opti_jnts(1)+opti_jnts(3))),
            # 1_left leg root   18_HIP_LEFT
            0.2*opti_jnts(25)+0.4*(opti_jnts(0)+opti_jnts(2)),
            # 2_right leg root  22_HIP_RIGHT
            0.2*opti_jnts(31)+0.4*(opti_jnts(1)+opti_jnts(3)),
            # 3_lowerback       1_SPINE_NAVAL
            0.25*(opti_jnts(0)+opti_jnts(1)+opti_jnts(2)+opti_jnts(3)),
            # 4_left knee       19_KNEE_LEFT
            opti_jnts(25),
            # 5_right knee      23_KNEE_RIGHT
            opti_jnts(31),
            # 6_upperback       0.75*2_SPINE_CHEST+0.25*1_SPINE_NAVAL
            0.25*(0.25*(opti_jnts(0)+opti_jnts(1)+opti_jnts(2)+opti_jnts(3)))+0.75*(0.5*(0.25*(opti_jnts(0)+opti_jnts(1)+opti_jnts(2)+opti_jnts(3))+0.5*(0.4 * opti_jnts(6) + 0.3 * opti_jnts(5) + 0.3 * opti_jnts(12)+0.4 * opti_jnts(7) + 0.3 * opti_jnts(5) + 0.3 * opti_jnts(4)))),
            # 7_left ankle      20_ANKLE_LEFT
            opti_jnts(27),
            # 8_right ankle     24_ANKLE_RIGHT
            opti_jnts(33),
            # 9_thorax          2_SPINE_CHEST
            0.5*(0.25*(opti_jnts(0)+opti_jnts(1)+opti_jnts(2)+opti_jnts(3))+0.5*(0.4 * opti_jnts(6) + 0.3 * opti_jnts(5) + 0.3 * opti_jnts(12)+0.4 * opti_jnts(7) + 0.3 * opti_jnts(5) + 0.3 * opti_jnts(4))),
            # 10_left toes      21_FOOT_LEFT
            0.5*opti_jnts(28)+0.5*opti_jnts(29),
            # 11_right toes     25_FOOT_RIGHT
            0.5*opti_jnts(35)+0.5*opti_jnts(36),
            # 12_lowerneck      3_NECK
            0.5*opti_jnts(4)+0.5*opti_jnts(5),
            # 13_left clavicle  0.4*4_CLAVICLE_LEFT+0.3*2_SPINE_CHEST+0.3*5_SHOULDER_LEFT
            0.4 * opti_jnts(6) + 0.3 * opti_jnts(5) + 0.3 * opti_jnts(12),
            # 14_right clavicle 0.4*11_CLAVICLE_RIGHT+0.3*2_SPINE_CHEST+0.3*12_SHOULDER_RIGHT
            0.4 * opti_jnts(7) + 0.3 * opti_jnts(5) + 0.3 * opti_jnts(4),
            # 15_upperneck      26_HEAD
            0.2*opti_jnts(8)+0.8*(0.5*opti_jnts(4)+0.5*opti_jnts(5)),
            # 16_left armroot   0.9*5_SHOULDER_LEFT+0.1*4_CLAVICLE_LEFT
            opti_jnts(12),
            # 17_right armroot  0.9*12_SHOULDER_RIGHT+0.1*11_CLAVICLE_RIGHT
            opti_jnts(19),
            # 18_left elbow     6_ELBOW_LEFT
            opti_jnts(13),
            # 19_right elbow  13_ELBOW_RIGHT
            opti_jnts(20),
            # 20_left wrist     7_WRIST_LEFT
            0.5*opti_jnts(16)+0.5*opti_jnts(17),
            # 21_right wrist    14_WRIST_RIGHT
            0.5*opti_jnts(23)+0.5*opti_jnts(24),
            # 22_left hand      8_HAND_LEFT
            opti_jnts(15),
            # 23_right hand     15_HAND_RIGHT
            opti_jnts(22),
            # ext_left finger tip   9_HANDTIP_LEFT
            2*opti_jnts(15)-0.5*opti_jnts(16)-0.5*opti_jnts(17),
            # ext_right finger tip  16_HANDTIP_RIGHT
            2*opti_jnts(22)-0.5*opti_jnts(23)-0.5*opti_jnts(24),
            # ext_left toe tip      21_FOOT_LEFT+0.3*(21_FOOT_LEFT-20_ANKLE_LEFT)
            0.5*opti_jnts(28)+0.5*opti_jnts(29) + 0.3 * (0.5*opti_jnts(28)+0.5*opti_jnts(29) - opti_jnts(27)),
            # ext_right toe tip     25_FOOT_RIGHT+0.3*(25_FOOT_RIGHT-24_ANKLE_RIGHT)
            0.5*opti_jnts(34)+0.5*opti_jnts(35) + 0.3 * (0.5*opti_jnts(34)+0.5*opti_jnts(35) - opti_jnts(33)),
            # ext_head_top          27_NOSE+1.5*((28_EYE_LEFT-27_NOSE)+(30_EYE_RIGHT-27_NOSE))
            opti_jnts(8)
        ])
        return self.smpl_jnts

    def normalization(self, jnts=None, pcl=None):
        if jnts is None:
            jnts = self.smpl_jnts
        if pcl is None:
            pcl = self.pcl
        # x_norm = np.linalg.norm(jnts, axis = 0, keepdims = True, ord=np.inf)
        self.scale = np.max(np.abs(jnts))
        x_norm = np.repeat(self.scale, 3)
        self.smpl_jnts = jnts / x_norm
        self.pcl = pcl/x_norm
        #self.smpl_jnts,self.pcl=self.smpl_joints_transfer_coordinates(self.smpl_jnts)
        return self.smpl_jnts, self.pcl

    def revert_transform(self):
        return np.linalg.inv(self.R.T), -self.t, self.scale

    def save_revert_transform(self, file_path):
        R, t, scale = self.revert_transform()
        np.savez(file_path, R=R, t=t, scale=scale)


def included_cos(a,b):
    return a/math.sqrt(a*a+b*b)
