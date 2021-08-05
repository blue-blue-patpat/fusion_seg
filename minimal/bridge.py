import numpy as np
import math


class JointsBridge():
    def __init__(self) -> None:
        self.k_jnt = np.zeros(28)

    def load_kinect_joints(self, kinect_joints):
        self.k_jnt = kinect_joints

    def update_smpl_joints(self):
        # see also: https://docs.microsoft.com/bs-latn-ba/azure/kinect-dk/body-joints#joint-coordinates
        self.smpl_jnt = np.array([
            # SMPL              Kinect
            # 0_pelvis          0.5*0_PELVIS+0.5*1_SPINE_NAVAL
            0.5 * self.k_jnt[0] + 0.5 * self.k_jnt[1],
            # 1_left leg root   18_HIP_LEFT
            self.k_jnt[18],
            # 2_right leg root  22_HIP_RIGHT
            self.k_jnt[22],
            # 3_lowerback       1_SPINE_NAVAL
            self.k_jnt[1],
            # 4_left knee       19_KNEE_LEFT
            self.k_jnt[19],
            # 5_right knee      23_KNEE_RIGHT
            self.k_jnt[23],
            # 6_upperback       0.75*2_SPINE_CHEST+0.25*1_SPINE_NAVAL
            0.75 * self.k_jnt[2] + 0.25 * self.k_jnt[1],
            # 7_left ankle      20_ANKLE_LEFT
            self.k_jnt[20],
            # 8_right ankle     24_ANKLE_RIGHT
            self.k_jnt[24],
            # 9_thorax          2_SPINE_CHEST
            self.k_jnt[2],
            # 10_left toes      21_FOOT_LEFT
            self.k_jnt[21],
            # 11_right toes     25_FOOT_RIGHT
            self.k_jnt[25],
            # 12_lowerneck      3_NECK
            self.k_jnt[3],
            # 13_left clavicle  0.4*4_CLAVICLE_LEFT+0.3*2_SPINE_CHEST+0.3*5_SHOULDER_LEFT
            0.4*self.k_jnt[4]+0.3*self.k_jnt[2]+0.3*self.k_jnt[5],
            # 14_right clavicle 0.4*11_CLAVICLE_RIGHT+0.3*2_SPINE_CHEST+0.3*12_SHOULDER_RIGHT
            0.4*self.k_jnt[11]+0.3*self.k_jnt[2]+0.3*self.k_jnt[12],
            # 15_upperneck      26_HEAD
            self.k_jnt[26],
            # 16_left armroot   0.9*5_SHOULDER_LEFT+0.1*4_CLAVICLE_LEFT
            0.9*self.k_jnt[5]+0.1*self.k_jnt[4],
            # 17_right armroot  0.9*12_SHOULDER_RIGHT+0.1*11_CLAVICLE_RIGHT
            0.9*self.k_jnt[12]+0.1*self.k_jnt[11],
            # 18_left elbow     6_ELBOW_LEFT
            self.k_jnt[6],
            # 19_right armroot  13_ELBOW_RIGHT
            self.k_jnt[13],
            # 20_left wrist     7_WRIST_LEFT
            self.k_jnt[7],
            # 21_right wrist    14_WRIST_RIGHT
            self.k_jnt[14],
            # 22_left hand      8_HAND_LEFT
            self.k_jnt[8],
            # 23_right hand     15_HAND_RIGHT
            self.k_jnt[15],
            # ext_left finger tip   9_HANDTIP_LEFT
            self.k_jnt[9],
            # ext_right finger tip  16_HANDTIP_RIGHT
            self.k_jnt[16],
            # ext_left toe tip      21_FOOT_LEFT+0.3*(21_FOOT_LEFT-20_ANKLE_LEFT)
            self.k_jnt[21] + 0.3*(self.k_jnt[21]-self.k_jnt[20]),
            # ext_right toe tip     25_FOOT_RIGHT+0.3*(25_FOOT_RIGHT-24_ANKLE_RIGHT)
            self.k_jnt[25] + 0.3*(self.k_jnt[25]-self.k_jnt[24]),
            # ext_head_top          27_NOSE+1.5*((28_EYE_LEFT-27_NOSE)+(30_EYE_RIGHT-27_NOSE))
            self.k_jnt[27] + 1.5*((self.k_jnt[28]-self.k_jnt[27]) + (self.k_jnt[30]-self.k_jnt[27]))
        ])
        return self.smpl_jnt


def KinectStoSMPL(file):    
    content= np.load(file)
    content=content[0,:,0:3]    #提取前三列

    a=content[0,0]  #前三列单独拿出来
    b=content[0,1]
    c=content[0,2]

    d=content[:,0:1]-a  #进行平移工作
    e=content[:,1:2]-b
    f=content[:,2:3]-c
    content[:,0:1]=d
    content[:,1:2]=e
    content[:,2:3]=f
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

    j=[[cos(i[1], i[0]), cos(i[0],i[1]),0], #绕z旋转矩阵
        [-cos(i[0],i[1]), cos(i[1],i[0]), 0],
        [0, 0, 1]]

    k=np.dot(h,j)
    l=np.dot(content,k)

    m=[[cos(l[18,0],l[18,2]),0,-cos(l[18,2],l[18,0])],  #绕y旋转矩阵
        [0,1,0],
        [cos(l[18,2],l[18,0]),0,cos(l[18,0],l[18,2])]]

    n=np.dot(l,m)   #乘到最后是前三列最终

    content=np.load(file)  #把前三列存回去
    content[:,:,0:3]=n
    return content
    '''
    pt2=open3d.geometry.PointCloud()
    pt2.points=open3d.utility.Vector3dVector(n.reshape(-1, 3))
    pt2.paint_uniform_color([0, 1, 0])

    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
    open3d.visualization.draw_geometries([point_cloud1,pt2]+[axis_pcd])
    '''