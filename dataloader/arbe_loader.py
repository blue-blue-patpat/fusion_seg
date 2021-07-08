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
    :param filepath: .bag file path
    :return: [idx, global_ts, other_fields] pandas.DataFrame, same idx for same frame.
    """
    bag_data = arbe_readfile_offline(filepath)

    clm = None
    df = pd.DataFrame()
    frame_idx = 0
    for topic, msg, t in bag_data:
        if clm is None:
            clm = ["frame_idx", "global_ts"] + [item.name for item in msg.fields]
            df = pd.DataFrame(columns=clm)
        gen = point_cloud2.read_points(msg)
        for p in gen:
            df.loc[len(df)] = [frame_idx, t] + list(p)
        frame_idx += 1

        print("arbe loader: {} frames saved.")
    return df


def arbe_loader_callback(msg, args):
    """
    ros data to dataframe
    :param data: vrpn message
    :param args: dict(frame_id, dataframe, name)
    :return: None
    """
    # print(data, args)
    ts = rospy.get_time()
    df = args.get('dataframe', pd.DataFrame())
    if df.empty:
        args["column"] = ["frame_id", "global_ts"] + [item.name for item in msg.fields]
        # df[["frame_id", "global_ts"]+[item.name for item in msg.fields]] = None
        df[args["column"]] = None
        
    #     args['frame_id'] = 0
    # else:
    #     args['frame_id'] += 1
    
    args['msg_list'][ts] = msg
    # for p in gen:
    #     df.loc[len(df)] = [args['frame_id'], ts] + list(p)
    print("{} frame saved at {}, total {}.".format(args.get('name', 'Anomaly'), ts, len(args['msg_list'])))


def arbe_loader_before_start(sub: dict):
    sub["args"].update(dict(name=sub["name"], dataframe=pd.DataFrame(), frame_id=0, msg_list={}, arr=[]))


def arbe_loader_after_stop(sub: dict):
    for ts, msg in sub["args"]['msg_list'].items():
        gen = point_cloud2.read_points(msg)
        start_t = time.time()
        last_t = start_t
        for p in gen:
            s = time.time()

            row_data = [sub["args"]["frame_id"], ts]
            row_data += list(p)

            sub['args']['arr'].append(row_data)
            d = time.time()
            print(d-s, d - last_t)
            last_t = d
        end_t = time.time()
        print("{} frame saved at {}, total {}, time period {}.".format(sub["args"]["frame_id"], ts, len(sub["args"]["arr"]), end_t - start_t))
        sub["args"]["frame_id"] += 1
    sub["args"]["dataframe"] = pd.DataFrame(sub['args']['arr'], columns=sub["args"]["column"])