#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   nokov_loader.py    
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/28 23:21    wxy        1.0         NOKOV dataloader
"""

import pandas as pd
# import lib
from geometry_msgs.msg import PoseStamped
# from tf.transformations import *

from dataloader.ros_client import MultiSubClient


"""
Use NOKOV Seeker to save offline data.
"""


def nokov_loader_callback(data, args):
    """
    NOKOV vrpn data to dataframe
    :param data: vrpn data
    :param args:
    :return: None
    """
    # print(data, args)
    df = args.get('df', pd.DataFrame())
    topic, msg, t = data
    if df.empty:
        df[["global_ts"]+[item.name for item in msg.fields]] = None
    # TODO: check df reference; check data structure
    df.loc[len(df)] = [msg.header.stamp, msg.pose.position.x, msg.pose.position.y,
                       msg.pose.position.z]
    print("nokov frame{} saved.".format(msg.header.frame_id))


def nokov_loader_before_start(sub: dict) -> dict:
    return dict(name=sub["name"], dataframe=pd.DataFrame())