#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   arbe_loader.py
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/28 22:34    wxy        1.0         arbe dataloader
"""

# import lib
from __future__ import generators
import time
import pandas as pd
import rosbag
import rospy
from sensor_msgs import point_cloud2


def arbe_readfile_offline(filepath: str, ):
    """
    Load arbe offline data from bag file

    :param filepath: .bag file path
    :return: data generator: (topic, msg, ts)
    """
    bag_file = filepath
    bag = rosbag.Bag(bag_file, "r")
    info = bag.get_type_and_topic_info()
    bag_data = bag.read_messages()
    print("Loading Arbe data, {}".format(info))
    return bag_data


def arbe_loader_offline(filepath: str) -> pd.DataFrame:
    """
    Load arbe offline data from bag file to pandas.DataFrame
    WARNING: need test.

    :param filepath: .bag file path
    :return: [idx, global_ts, other_fields] pandas.DataFrame, same idx for same frame.
    """
    bag_data = arbe_readfile_offline(filepath)

    dataframes = {}
    for topic, msg, t in bag_data:
        dataframes[t] =  _msg_to_dataframe(msg)
        # if clm is None:
        #     clm = ["frame_idx", "global_ts"] + [item.name for item in msg.fields]
        #     df = pd.DataFrame(columns=clm)
        # gen = point_cloud2.read_points(msg)
        # for p in gen:
        #     df.loc[len(df)] = [frame_idx, t] + list(p)
        # frame_idx += 1

        print("arbe loader: {} frames saved.")
    return dataframes


# def arbe_loader_before_start_abandoned(sub: dict):
#     sub["args"].update(dict(name=sub["name"], dataframe=pd.DataFrame(), frame_id=0, msg_list={}, arr=[]))


def arbe_loader_before_start(sub: dict):
    """
    MultiSubClient before subscriber start trigger
    Init args

    :param sub: current subscriber
    """
    sub["args"].update(dict(name=sub["name"], dataframe={}, task_queue={}))


def arbe_loader_callback(msg, args):
    """
    ros data stream to dataframe

    :param data: vrpn message
    :param args: dict(dataframe, name, task_queue)
    :return: None
    """
    ts = rospy.get_time()
    
    if args.get('force_realtime', True):
        args["dataframe"][ts] = _msg_to_dataframe(msg)
    else:
        args["task_queue"][ts] = msg
    print("[{}] {} frames saved, {} frames waiting: {}".format(
        args['name'], len(args["dataframe"]), len(args['task_queue']), ts))


# def arbe_loader_after_stop_abandoned(sub: dict):
#     for ts, msg in sub["args"]['msg_list'].items():
#         gen = point_cloud2.read_points(msg)
#         start_t = time.time()
#         last_t = start_t
#         for p in gen:
#             s = time.time()

#             row_data = [sub["args"]["frame_id"], ts]
#             row_data += list(p)

#             sub['args']['arr'].append(row_data)
#             d = time.time()
#             print(d-s, d - last_t)
#             last_t = d
#         end_t = time.time()
#         print("{} frame saved at {}, total {}, time period {}.".format(sub["args"]["frame_id"], ts, len(sub["args"]["arr"]), end_t - start_t))
#         sub["args"]["frame_id"] += 1
#     sub["args"]["dataframe"] = pd.DataFrame(sub['args']['arr'], columns=sub["args"]["column"])


def arbe_loader_after_stop(sub: dict):
    """
    MultiSubClient after subscriber stop trigger
    Convert data in args["task_queue"]

    :param sub: current subscriber
    """
    for ts, msg in sub["args"]['task_queue'].items():
        sub["args"]["dataframe"][ts] = _msg_to_dataframe(msg)
        

def _msg_to_dataframe(msg) -> pd.DataFrame:
    """
    Convert ros message to pandas DataFrame

    :param msg: ros messgae
    :return: DataFrame
    """
    start_t = time.time()
    df = pd.DataFrame(_point_cloud_loader(msg), columns=[item.name for item in msg.fields])
    end_t = time.time()
    print("{} points, time period {}.".format(len(df), end_t - start_t))
    return df


def _point_cloud_loader(cloud):
    """
    Implementation of point_cloud2.read_points
    Return list directly, rather than Generator to speed up stream read.

    :param sub: current subscriber
    """
    import struct
    fmt = point_cloud2._get_struct_fmt(cloud.is_bigendian, cloud.fields, None)
    width, height, point_step, row_step, data = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data
    unpack_from = struct.Struct(fmt).unpack_from
    ret = []

    for v in range(height):
        offset = row_step * v
        for u in range(width):
            ret.append(unpack_from(data, offset))
            offset += point_step
    return ret
