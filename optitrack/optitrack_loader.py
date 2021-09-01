import pandas
import numpy as np
import time
import os

def parse_opti_csv(csv_path):
    with open(csv_path, 'r') as f:
        device_data = f.readline().split(',')
    df = pandas.read_csv(csv_path, skiprows=2)
    file_path, _ = os.path.split(csv_path)
    # transform time to timestamp
    start_time = time.mktime(time.strptime(device_data[11].replace("上午", "am").replace("下午", "pm"), "%Y-%m-%d %I.%M.%S.%f %p"))
    start_time += float(device_data[11][-6:-3])/1000.
    # extract the bone_markers and bones dataframe
    markers_df = df.iloc[4:,[i for i, item in enumerate(df.columns) if "Bone Marker" in item]]
    bones_df = df.iloc[4:,[i for i, item in enumerate(df.columns) if "Bone Marker" not in item and "Bone" in item]]
    num_of_frames = len(markers_df)

    markers_array = np.asarray(markers_df, np.float64).reshape(num_of_frames,-1,3)
    bones_array = np.asarray(bones_df, np.float64).reshape(num_of_frames,-1,7)

    for i in range(num_of_frames):
        tm = start_time + float(df.iloc[i+4, 1])
        filename = "{}/id={}_st={}".format(file_path, i, tm)
        np.savez(filename, markers=markers_array[i].reshape(-1,37,3), bones=bones_array[i].reshape(-1,21,7))
