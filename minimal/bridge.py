import numpy as np


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
        