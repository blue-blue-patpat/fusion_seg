import pandas
import numpy as np
import time
import os

def csv_parser(csv_path):
    df = pandas.read_csv(csv_path)
    # df.fillna('0')
    file_path, _ = os.path.split(csv_path)
    # transform time to timestamp
    timestamp = time.mktime(time.strptime(df.columns[3][5:], "%Y-%m-%d %I.%M.%S %p"))
    start_time = timestamp + int(df.columns[11][-6:-3])/1000
    # extract the bone_markers and bones dataframe
    markers_df = df.iloc[6:,[i for i, item in enumerate(df.iloc[1]) if item == "Bone Marker"]]
    bones_df = df.iloc[6:,[i for i, item in enumerate(df.iloc[1]) if item == "Bone"]]
    num_of_frames = len(markers_df)

    markers_array = np.asarray(markers_df, np.float64).reshape(num_of_frames,-1,3)
    bones_array = np.asarray(bones_df, np.float64).reshape(num_of_frames,-1,7)

    for i in range(num_of_frames):
        tm = start_time + float(df.iloc[i+6, 1])
        filename = "{}/id={}_tm={}".format(file_path, i, tm)
        np.savez(filename, markers=markers_array[i].reshape(-1,37,3), bones=bones_array[i].reshape(-1,21,7))
