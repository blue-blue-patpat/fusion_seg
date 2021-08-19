import numpy as np
import json
from scipy.spatial.transform import Rotation as R

'''
本文件用于将三台Kinect坐标系转换到arbe坐标系，最终的齐次变换矩阵保存在npy文件中
使用时调用quaternion_matrix,输入json文件所在目录即可
这里需要注意，变换到arbe的平移矩阵是通过STL文件读取的，在实际显示时可能有些误差
可能需要微调
'''
def quaternion_matrix(file1,file2,file3):
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

    R_arbe = np.zeros((3,3))
    R_arbe[0,0] = 1
    R_arbe[1,2] = 1
    R_arbe[2,1] = -1
    T_arbe = np.array([0, 0.05, -0.2])
    final_R1 = R_arbe
    final_R2 = np.dot(np.dot(R_arbe,np.linalg.inv(rotation_matrix1)),rotation_matrix2)
    final_R3 = np.dot(np.dot(R_arbe,np.linalg.inv(rotation_matrix1)),rotation_matrix3)
    # double final_T1[3] = { 0, 0.05, -0.2 };
    # double final_T2[3] = { T4 - T1,T5 - T2 + 0.05,T6 - T3 - 0.2 };
    # double final_T3[3] = { T7 - T1,T8 - T2 + 0.05,T9 - T3 - 0.2 };
    H1 = np.column_stack((final_R1,T_arbe.T))
    H1 = np.row_stack((H1,np.array([0,0,0,1]).T))
    H2 = np.column_stack((final_R2,np.array([T4-T1,T5-T2,T6-T3]).T-T_arbe.T))
    H2 = np.row_stack((H2,np.array([0,0,0,1]).T))
    H3 = np.column_stack((final_R3,np.array([T7-T1,T8-T2,T9-T3]).T-T_arbe.T))
    H3 = np.row_stack((H3,np.array([0,0,0,1]).T))
    np.save("ignoredata/calib_files/kinect1.npy",H1)
    np.save("ignoredata/calib_files/kinect2.npy",H2)
    np.save("ignoredata/calib_files/kinect3.npy",H3)

if __name__ == "__main__":
    file1 = '/home/nesc525/chen/kinect/kinect_calib/k4a-calibration-master/build/cn01.json'
    file2 = '/home/nesc525/chen/kinect/kinect_calib/k4a-calibration-master/build/cn02.json'
    file3 = '/home/nesc525/chen/kinect/kinect_calib/k4a-calibration-master/build/cn03.json'
    quaternion_matrix(file1,file2,file3)
    print(np.load("ignoredata/calib_files/kinect2.npy"))