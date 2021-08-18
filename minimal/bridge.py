import numpy as np
import math


class JointsBridge():
    def __init__(self) -> None:
        self.k_jnts = np.zeros((32, 3))
        self.smpl_jnts = np.zeros((28, 3))

    def smpl_from_kinect(self, jnts, pcl=np.array([[0,0,0]])):
        self.kinect_joints_transfer_coordinates(jnts, pcl)
        self.update_smpl_joints_from_kinect_joints()
        return self.normalization()

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
        t = -np.array([0.5 * content[0] + 0.5 * content[1]])

        # apply transfer to all joints
        content = content + np.repeat(t, content.shape[0], 0)

        '''
        point_cloud1 = open3d.geometry.PointCloud()
        point_cloud1.points = open3d.utility.Vector3dVector(content)
        point_cloud1.paint_uniform_color([1,0,0])
        '''
        def cos(a,b):
            return a/math.sqrt(a*a+b*b)

        g=content[1,:]  #旋转r1到y轴

        h=[[1, 0, 0],   #绕x旋转矩阵
            [0, cos(g[1],g[2]), -cos(g[2],g[1])],
            [0, cos(g[2],g[1]), cos(g[1],g[2])]]

        i=np.dot(g,h)

        j=[[cos(i[1], i[0]), -cos(i[0],i[1]),0], #绕z旋转矩阵
            [cos(i[0],i[1]), cos(i[1],i[0]), 0],
            [0, 0, 1]]

        k=np.dot(h,j)
        l=np.dot(content,k)

        m=[[cos(l[18,0],l[18,2]),0,cos(l[18,2],l[18,0])],  #绕y旋转矩阵
            [0,1,0],
            [-cos(l[18,2],l[18,0]),0,cos(l[18,0],l[18,2])]]
        
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
            0.75 * kinect_joints[2] + 0.25 * kinect_joints[1],
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
            kinect_joints[27] + 1.5*((kinect_joints[28]-kinect_joints[27]) + (kinect_joints[30]-kinect_joints[27]))
        ])
        return self.smpl_jnts

    def normalization(self, jnts=None, pcl=None):
        if jnts is None:
            jnts = self.smpl_jnts
        if pcl is None:
            pcl = self.pcl
        # x_norm = np.linalg.norm(jnts, axis = 0, keepdims = True, ord=np.inf)
        x_norm = np.repeat(np.max(jnts), 3)
        self.smpl_jnts = jnts / x_norm
        self.pcl = pcl/x_norm
        return self.smpl_jnts, self.pcl


