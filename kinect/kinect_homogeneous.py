import numpy as np
import json
from scipy.spatial.transform import Rotation as R

'''
本文件用于将三台Kinect坐标系转换到arbe坐标系，最终的齐次变换矩阵保存在npy文件中
使用时调用quaternion_matrix,输入json文件所在目录即可
这里需要注意，变换到arbe的平移矩阵是通过STL文件读取的，在实际显示时可能有些误差
可能需要微调
'''
def quaternion_matrix(file_path):
    file1 = file_path + "/cn01.json"
    file2 = file_path + "/cn02.json"
    file3 = file_path + "/cn03.json"
    with open(file1,'r',encoding='utf8')as fp1:
        json_data1 = json.load(fp1)
    x = json_data1['value0']['rotation']['x']
    y = json_data1['value0']['rotation']['y']
    z = json_data1['value0']['rotation']['z']
    w = json_data1['value0']['rotation']['w']
    T1 = json_data1['value0']['translation']['m00']
    T2 = json_data1['value0']['translation']['m10']
    T3 = json_data1['value0']['translation']['m20']
    Rq = [x,y,z,w]
    Rm = R.from_quat(Rq)
    rotation_matrix1 = Rm.as_matrix()

    with open(file2,'r',encoding='utf8')as fp2:
        json_data2 = json.load(fp2)
    x = json_data2['value0']['rotation']['x']
    y = json_data2['value0']['rotation']['y']
    z = json_data2['value0']['rotation']['z']
    w = json_data2['value0']['rotation']['w']
    T4 = json_data2['value0']['translation']['m00']
    T5 = json_data2['value0']['translation']['m10']
    T6 = json_data2['value0']['translation']['m20']
    Rq = [x,y,z,w]
    Rm = R.from_quat(Rq)
    rotation_matrix2 = Rm.as_matrix()

    with open(file3,'r',encoding='utf8')as fp3:
        json_data3 = json.load(fp3)
    x = json_data3['value0']['rotation']['x']
    y = json_data3['value0']['rotation']['y']
    z = json_data3['value0']['rotation']['z']
    w = json_data3['value0']['rotation']['w']
    T7 = json_data3['value0']['translation']['m00']
    T8 = json_data3['value0']['translation']['m10']
    T9 = json_data3['value0']['translation']['m20']
    Rq = [x,y,z,w]
    Rm = R.from_quat(Rq)
    rotation_matrix3 = Rm.as_matrix()
    #R_arbe是从arbe坐标系到master坐标系的旋转矩阵，T_arbe是从arbe坐标系到master坐标系的平移矩阵
    R_arbe = np.zeros((3,3))
    R_arbe[0,0] = 1
    R_arbe[1,2] = 1
    R_arbe[2,1] = -1
    T_arbe = np.array([0, 0.05, -0.2])
    final_R1 = R_arbe
    final_R1 = np.linalg.inv(final_R1)
    # final_R2是从arbe坐标系到sub1坐标系的旋转矩阵，这里因为从json读取的旋转矩阵都是相对于世界坐标系的，
    # 因此要求得从master到sub1的旋转矩阵，应该先对master的旋转矩阵求逆，再乘以sub1的旋转矩阵，得到从master到
    # sub1的旋转矩阵，再结合arbe到master2的旋转矩阵，就可以求出arbe到sub1的旋转矩阵
    final_R2 = np.dot(np.dot(R_arbe,np.linalg.inv(rotation_matrix1)),rotation_matrix2)
    final_R2 = np.linalg.inv(final_R2)
    # final_R3是从arbe坐标系到sub2坐标系的旋转矩阵，这里因为从json读取的旋转矩阵都是相对于世界坐标系的，
    # 因此要求得从master到sub2的旋转矩阵，应该先对master的旋转矩阵求逆，再乘以sub2的旋转矩阵，得到从master到
    # sub2的旋转矩阵，再结合arbe到master的旋转矩阵，就可以求出arbe到sub2的旋转矩阵
    final_R3 = np.dot(np.dot(R_arbe,np.linalg.inv(rotation_matrix1)),rotation_matrix3)
    final_R3 = np.linalg.inv(final_R3)
    # double final_T2[3] = { T4 - T1,T5 - T2 + 0.05,T6 - T3 - 0.2 };
    # double final_T3[3] = { T7 - T1,T8 - T2 + 0.05,T9 - T3 - 0.2 };
    H1 = np.column_stack((final_R1,T_arbe.T))
    H1 = np.row_stack((H1,np.array([0,0,0,1]).T))
    H2 = np.column_stack((final_R2,np.array([T4-T1,T5-T2,T6-T3]).T-T_arbe.T))
    H2 = np.row_stack((H2,np.array([0,0,0,1]).T))
    H3 = np.column_stack((final_R3,np.array([T7-T1,T8-T2,T9-T3]).T-T_arbe.T))
    H3 = np.row_stack((H3,np.array([0,0,0,1]).T))
    np.save("{}/trans_master.npy".format(file_path),H1)
    np.save("{}/trans_sub1.npy".format(file_path),H2)
    np.save("{}/trans_sub2.npy".format(file_path),H3)


if __name__ == "__main__":
    file_path = "/media/nesc525/perple/kinect_calib"
    quaternion_matrix(file_path)
    print(np.load(file_path + "/trans_sub1.npy"))