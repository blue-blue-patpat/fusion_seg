import numpy as np
import math


class JointsBridge():
    def __init__(self) -> None:
        self.jnts = None
        self.pcl = None

        self.R = np.zeros([3,3])
        self.t = np.zeros([1,3])
        self.scale = 1

    def init_input(self, jnts, pcl) -> None:
        self.jnts = jnts
        self.pcl = pcl

    def smpl_from_kinect(self):
        self.kinect_jnts_to_smpl()
        self.smpl_joints_transform()
        self.filter_pcl()
        self.normalization()
        return self.jnts, self.pcl

    def smpl_from_optitrack(self):
        self.optitrack_jnts_to_smpl()
        self.smpl_joints_transform()
        self.filter_pcl()
        self.normalization()
        return self.jnts, self.pcl

    def save_revert_transform(self, file_path):
        R, t, scale = self.revert_transform()
        np.savez(file_path, R=R, t=t, scale=scale)

    def filter_pcl(self) -> np.ndarray:
        upper_bound = self.jnts.max(axis=0) + 0.3
        lower_bound = self.jnts.min(axis=0) - 0.3
        # remove ground pcl
        lower_bound[1] += 0.32
        _filter = np.apply_along_axis(
            lambda row:\
                np.all((lower_bound<row) & (row<upper_bound))\
                and np.linalg.norm(self.jnts - row, axis=1, keepdims=True).min() < 0.25,\
        axis=1, arr=self.pcl)
        self.pcl = self.pcl[_filter]
    
    def kinect_joints_transform(self): 
        # translate
        self.t = -np.array([0.75 * self.jnts[2] + 0.25 * self.jnts[1]])

        # apply to all joints
        self.jnts = self.jnts + np.repeat(self.filter_pclt, self.jnts.shape[0], 0)

        g=(self.jnts[4,:]+self.jnts[11,:])/2  #旋转r1到y轴

        h=[[1, 0, 0],   #绕x旋转矩阵
            [0, included_cos(g[1],g[2]), -included_cos(g[2],g[1])],
            [0, included_cos(g[2],g[1]), included_cos(g[1],g[2])]]

        i=np.dot(g,h)

        j=[[included_cos(i[1], i[0]), -included_cos(i[0],i[1]),0], #绕z旋转矩阵
            [included_cos(i[0],i[1]), included_cos(i[1],i[0]), 0],
            [0, 0, 1]]

        k=np.dot(h,j)
        l=np.dot(self.jnts,k)

        m=[[included_cos(l[4,0],l[4,2]),0,included_cos(l[4,2],l[4,0])],  #绕y旋转矩阵
            [0,1,0],
            [-included_cos(l[4,2],l[4,0]),0,included_cos(l[4,0],l[4,2])]]
        
        self.jnts = np.dot(l,m)
        self.R = np.dot(k,m)

        self.pcl = np.dot(self.pcl + np.repeat(self.t, self.pcl.shape[0], 0), self.R)

    def smpl_joints_transform(self):
        self.t = -self.jnts[6] #坐标规范下6号点为原点

        # apply to all joints
        self.jnts = self.jnts + self.t

        g=self.jnts[9,:]  #旋转r1到y轴

        h=[[1, 0, 0],   #绕x旋转矩阵
            [0, included_cos(g[1],g[2]), -included_cos(g[2],g[1])],
            [0, included_cos(g[2],g[1]), included_cos(g[1],g[2])]]

        i=np.dot(g,h)

        j=[[included_cos(i[1], i[0]), -included_cos(i[0],i[1]),0], #绕z旋转矩阵
            [included_cos(i[0],i[1]), included_cos(i[1],i[0]), 0],
            [0, 0, 1]]

        k=np.dot(h,j)
        l=np.dot(self.jnts,k)

        m=[[included_cos(l[13,0],l[13,2]),0,included_cos(l[13,2],l[13,0])],  #绕y旋转矩阵
            [0,1,0],
            [-included_cos(l[13,2],l[13,0]),0,included_cos(l[13,0],l[13,2])]]
        
        self.jnts = np.dot(l,m)   #乘到最后是前三列最终
        self.R = np.dot(k,m)    #matrix是最终旋转矩阵
        self.pcl = np.dot(self.pcl + self.t, self.R)
        return self.jnts, self.pcl

    def kinect_jnts_to_smpl(self):
        # see also: https://docs.microsoft.com/bs-latn-ba/azure/kinect-dk/body-joints#joint-coordinates
        self.jnts = np.array([
            # SMPL              Kinect
            # 0_pelvis          0.5*0_PELVIS+0.5*1_SPINE_NAVAL
            0.5 * self.jnts[0] + 0.5 * self.jnts[1],
            # 1_left leg root   18_HIP_LEFT
            self.jnts[18],
            # 2_right leg root  22_HIP_RIGHT
            self.jnts[22],
            # 3_lowerback       1_SPINE_NAVAL
            self.jnts[1],
            # 4_left knee       19_KNEE_LEFT
            self.jnts[19],
            # 5_right knee      23_KNEE_RIGHT
            self.jnts[23],
            # 6_upperback       0.75*2_SPINE_CHEST+0.25*1_SPINE_NAVAL
            0.65 * self.jnts[2] + 0.35 * self.jnts[1],
            # 7_left ankle      20_ANKLE_LEFT
            self.jnts[20],
            # 8_right ankle     24_ANKLE_RIGHT
            self.jnts[24],
            # 9_thorax          2_SPINE_CHEST
            self.jnts[2],
            # 10_left toes      21_FOOT_LEFT
            self.jnts[21],
            # 11_right toes     25_FOOT_RIGHT
            self.jnts[25],
            # 12_lowerneck      3_NECK
            self.jnts[3],
            # 13_left clavicle  0.4*4_CLAVICLE_LEFT+0.3*2_SPINE_CHEST+0.3*5_SHOULDER_LEFT
            0.4*self.jnts[4]+0.3*self.jnts[2]+0.3*self.jnts[5],
            # 14_right clavicle 0.4*11_CLAVICLE_RIGHT+0.3*2_SPINE_CHEST+0.3*12_SHOULDER_RIGHT
            0.4*self.jnts[11]+0.3*self.jnts[2]+0.3*self.jnts[12],
            # 15_upperneck      26_HEAD
            self.jnts[26],
            # 16_left armroot   0.9*5_SHOULDER_LEFT+0.1*4_CLAVICLE_LEFT
            0.9*self.jnts[5]+0.1*self.jnts[4],
            # 17_right armroot  0.9*12_SHOULDER_RIGHT+0.1*11_CLAVICLE_RIGHT
            0.9*self.jnts[12]+0.1*self.jnts[11],
            # 18_left elbow     6_ELBOW_LEFT
            self.jnts[6],
            # 19_right armroot  13_ELBOW_RIGHT
            self.jnts[13],
            # 20_left wrist     7_WRIST_LEFT
            self.jnts[7],
            # 21_right wrist    14_WRIST_RIGHT
            self.jnts[14],
            # 22_left hand      8_HAND_LEFT
            self.jnts[8],
            # 23_right hand     15_HAND_RIGHT
            self.jnts[15],
            # ext_left finger tip   9_HANDTIP_LEFT
            self.jnts[9],
            # ext_right finger tip  16_HANDTIP_RIGHT
            self.jnts[16],
            # ext_left toe tip      21_FOOT_LEFT+0.3*(21_FOOT_LEFT-20_ANKLE_LEFT)
            self.jnts[21] + 0.3*(self.jnts[21]-self.jnts[20]),
            # ext_right toe tip     25_FOOT_RIGHT+0.3*(25_FOOT_RIGHT-24_ANKLE_RIGHT)
            self.jnts[25] + 0.3*(self.jnts[25]-self.jnts[24]),
            # ext_head_top          27_NOSE+1.5*((28_EYE_LEFT-27_NOSE)+(30_EYE_RIGHT-27_NOSE))
            self.jnts[3] + 1*((self.jnts[29]-self.jnts[3]) + (self.jnts[31]-self.jnts[3]))
        ])

    def optitrack_jnts_to_smpl(self):
        self.jnts = np.array([
            # SMPL              OptiTrack
            # 0_pelvis          middle of 0：waist left front, 1: waist right front, 2: wrist left back and 3: wrist right back
            0.25 * (self.jnts[0] + self.jnts[1] + self.jnts[2] + self.jnts[3]),
            # 1_left leg root   1/4 centre of 25: left knee out and the centre of 0：waist left front, 2: wrist left back, and reduce the 1/4 of people's waistline
            0.2*self.jnts[25]+0.4*(self.jnts[0]+self.jnts[2])-0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 2_right leg root  1/4 centre of 31: right knee out and the centre of 1：waist right front, 3: wrist right back, and reduce the 1/4 of people's waistline
            0.2*self.jnts[31]+0.4*(self.jnts[1]+self.jnts[3])+0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 3_lowerback       middle of 0_pelvis and 6_upperback
            0.5*(0.25 * (self.jnts[0] + self.jnts[1] + self.jnts[2] + self.jnts[3])+0.5 * (0.25 * (self.jnts[0] + self.jnts[1] + self.jnts[2] + self.jnts[3]) + 0.5 * (0.4 * self.jnts[6] + 0.3 * self.jnts[5] + 0.3 * self.jnts[12] + 0.4 * self.jnts[7] + 0.3 * self.jnts[5] + 0.3 * self.jnts[4]))),
            # 4_left knee       25: left knee out, and reduce the 1/4 of people's waistline
            self.jnts[25]-0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 5_right knee      31: right knee out, and reduce the 1/4 of people's waistline
            self.jnts[31]+0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 6_upperback       middle of 0_pelvis and clavicle(middle of left clavicle and right clavicle)
            0.5 * (0.25 * (self.jnts[0] + self.jnts[1] + self.jnts[2] + self.jnts[3]) + 0.5 * (0.4 * self.jnts[6] + 0.3 * self.jnts[5] + 0.3 * self.jnts[12] + 0.4 * self.jnts[7] + 0.3 * self.jnts[5] + 0.3 * self.jnts[4])),
            # 7_left ankle      27: left ankle out, and reduce the 1/4 of people's waistline
            self.jnts[27]-0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 8_right ankle     33: right ankle out, and reduce the 1/4 of people's waistline
            self.jnts[33]+0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 9_thorax          the 1/4 centre of 6_upperback and clavicle(middle of left clavicle and right clavicle)
            0.75*(0.5 * (0.25 * (self.jnts[0] + self.jnts[1] + self.jnts[2] + self.jnts[3]) + 0.5 * (0.4 * self.jnts[6] + 0.3 * self.jnts[5] + 0.3 * self.jnts[12] + 0.4 * self.jnts[7] + 0.3 * self.jnts[5] + 0.3 * self.jnts[4])))+0.125*(0.4 * self.jnts[6] + 0.3 * self.jnts[5] + 0.3 * self.jnts[12]+0.4 * self.jnts[7] + 0.3 * self.jnts[5] + 0.3 * self.jnts[4]),
            # 10_left toes      the middle of 28: leftshin and 29: LToeOut and reduce the 1/4 of people's waistline
            0.5*self.jnts[28]+0.5*self.jnts[29]-0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 11_right toes     the middle of 35: right toe out and 36: right toe in and reduce the 1/4 of people's waistline
            0.5*self.jnts[35]+0.5*self.jnts[36]+0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # 12_lowerneck      4: back top add 1/2 of waist front reduce back
            self.jnts[4]+0.25*(self.jnts[0]+self.jnts[1]-self.jnts[2]-self.jnts[3]),
            # 13_left clavicle  0.4*6: back left+0.3*5: chest+0.3*12: left shoulder top
            0.4 * self.jnts[6] + 0.3 * self.jnts[5] + 0.3 * self.jnts[12],
            # 14_right clavicle 0.4*7: back right+0.3*5: chest+0.3*19: right shoulder top
            0.4 * self.jnts[7] + 0.3 * self.jnts[5] + 0.3 * self.jnts[19],
            # 15_upperneck      1/5 centre of 8: head top and 4: back top
            0.2*self.jnts[8]+0.8*self.jnts[4],
            # 16_left armroot   1/4 centre of 12: left shoulder top and 13: left elbow out
            0.75*self.jnts[12]+0.25*self.jnts[13]+0.5*self.jnts[4]+0.5*self.jnts[5]-0.2*self.jnts[8]-0.8*(0.5*self.jnts[4]+0.5*self.jnts[5]),
            # 17_right armroot  1/4 centre of 19: right shoulder top and 20: right elbow out
            0.75*self.jnts[19]+0.25*self.jnts[20]+0.5*self.jnts[4]+0.5*self.jnts[5]-0.2*self.jnts[8]-0.8*(0.5*self.jnts[4]+0.5*self.jnts[5]),
            # 18_left elbow     13: left elbow out
            self.jnts[13],
            # 19_right elbow    20: right elbow out
            self.jnts[20],
            # 20_left wrist     the middle of 16: left wrist out and 17: left wrist in
            0.5*self.jnts[16]+0.5*self.jnts[17],
            # 21_right wrist    the middle of 23: right wrist out and 24: right wrist in
            0.5*self.jnts[23]+0.5*self.jnts[24],
            # 22_left hand      15: left hand out
            self.jnts[15],
            # 23_right hand     22: right hand out
            self.jnts[22],
            # ext_left finger tip   22_left hand+22_left hand-20_left wrist
            2*self.jnts[15]-0.5*self.jnts[16]-0.5*self.jnts[17],
            # ext_right finger tip  23_right hand+23_right hand-21_right wrist
            2*self.jnts[22]-0.5*self.jnts[23]-0.5*self.jnts[24],
            # ext_left toe tip  based on 10_left toes and add the length
            0.5*self.jnts[28]+0.5*self.jnts[29] + 0.3 * (0.5*self.jnts[28]+0.5*self.jnts[29] - self.jnts[27])-0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # ext_right toe tip based on 11_right toes and add the length
            0.5*self.jnts[34]+0.5*self.jnts[35] + 0.3 * (0.5*self.jnts[34]+0.5*self.jnts[35] - self.jnts[33])+0.125*(self.jnts[0]-self.jnts[1]+self.jnts[2]-self.jnts[3]),
            # ext_head_top      8: head top
            self.jnts[8]
        ])

    def normalization(self):
        # x_norm = np.linalg.norm(jnts, axis = 0, keepdims = True, ord=np.inf)
        self.scale = np.max(np.abs(self.jnts))
        x_norm = np.repeat(self.scale, 3)
        self.jnts = self.jnts / x_norm
        self.pcl = self.pcl/x_norm

    def revert_transform(self):
        """
        Return revert transform matrix R, t, scale
        To transform back to origin coordinates, use (R @ array(3, n) + t) * scale
        """
        return np.linalg.inv(self.R.T), -self.t/self.scale, self.scale


def included_cos(a,b):
    return a/math.sqrt(a*a+b*b)
