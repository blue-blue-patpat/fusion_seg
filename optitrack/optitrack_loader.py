import pandas
import numpy as np
import time
import os

def csv_parser(csv_path):
    df = pandas.read_csv(csv_path)
    file_path, _ = os.path.split(csv_path)
    timestamp = time.mktime(time.strptime(df.columns[3][5:], "%Y-%m-%d %I.%M.%S %p"))
    start_time = timestamp + int(df.columns[11][-6:-3])/1000
    bone_markers = df.iloc[6:,[i for i, item in enumerate(df.iloc[1]) if item == "Bone Marker"]]
    bones = df.iloc[6:,[i for i, item in enumerate(df.iloc[1]) if item == "Bone"]]
    # names = []
    bones_list = []
    _array = np.asarray(bone_markers.iloc[:, :], np.float64).reshape(-1,37,3)
    bone_markers_array = np.pad(_array, ((0,0),(0,0),(4,0)), 'constant', constant_values = (0,0))
    bones_array = np.asarray(bones.iloc[:, :], np.float64).reshape(-1,21,7)
    for i in range(len(bone_markers_array)):
        bones_list.append(np.vstack((bone_markers_array[i], bones_array[i])))
        tm = start_time + float(df.iloc[i+6, 1])
        filename = "{}/id={}_tm={}".format(file_path, i, tm)
        np.save(filename, np.asarray(bones_list).reshape(-1,58,7))


if __name__ == "__main__":
    csv_parser("/home/nesc525/chen/3DSVC/ignoredata/optitrack_files/Take 2021-08-17 09.24.57 PM.csv")