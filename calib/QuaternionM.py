import numpy as np
import json
from scipy.spatial.transform import Rotation as R


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

    final_R2 = rotation_matrix2 * np.linalg.inv(rotation_matrix1)
    final_R3 = rotation_matrix3 * np.linalg.inv(rotation_matrix1)

    R2 = np.array([[0,0,0,T4-T1],
          [0,0,0,T5-T2],
          [0,0,0,T6-T3],
          [0,0,0,1]])
    R2[:3,:3] = final_R2

    R3 = np.array([[0, 0, 0, T7 - T1],
                   [0, 0, 0, T8 - T2],
                   [0, 0, 0, T9 - T3],
                   [0, 0, 0, 1]])
    R3[:3, :3] = final_R3

    result=[R2,R3]
    np.save(file1[:-5]+'.npy',result)
    #matrix=np.load(file1[:-5]+'.npy')
    #print(matrix)

file1 = './data/cn01.json'
file2 = './data/cn02.json'
file3 = './data/cn03.json'
QuaternionMatrix(file1,file2,file3)