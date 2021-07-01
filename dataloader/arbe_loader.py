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
import pandas as pd
import rosbag
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
