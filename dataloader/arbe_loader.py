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
import pandas as pd
import rosbag
from sensor_msgs import point_cloud2


def arbe_readfile_offline(filepath: str, ) -> list:
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
    idx = 0
    for topic, msg, t in bag_data:
        if clm is None:
            # TODO: check clm datatype
            clm = ["idx", "global_ts"] + list(msg.fields)
        print(msg.fields)

        # for i in msg.data:
        #     data.append(i)
        # print(data)
        gen = point_cloud2.read_points(msg)
        # print(type(gen))
        # print(gen[0], gen[1])
        for p in gen:
            # TODO: check p datatype
            df = df.append(pd.DataFrame([[idx, t] + list(p)], columns=clm))
            print(p)
        idx += 1

        print(t)
        print("--------------------------")
    return df
