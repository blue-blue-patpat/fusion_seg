#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   nokov_loader.py    
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/28 23:21    wxy        1.0         NOKOV dataloader
"""

# import lib
import pandas as pd
import rospy
from geometry_msgs.msg import PoseStamped
# from tf.transformations import *


"""
Use NOKOV Seeker to save offline data.
"""


def nokov_loader_before_start(sub: dict):
    """
    MultiSubClient before subscriber start trigger
    Init args

    :param sub: current subscriber
    """
    sub["args"].update(dict(name=sub["name"], dataframe={}, task_queue={}))


def nokov_loader_callback(msg, args):
    """
    NOKOV vrpn data to dataframe
    :param data: vrpn data
    :param args: subscriber["args"]
    :return: None
    """
    ts = rospy.get_time()

    if args.get('force_realtime', True):
        args["dataframe"][ts] = _msg_to_dataframe(msg)
    else:
        args["task_queue"][ts] = msg
    print("[{}] {} frames saved, {} frames waiting: {}".format(
        args['name'], len(args["dataframe"]), len(args['task_queue']), ts))


def nokov_loader_after_stop(sub: dict):
    """
    MultiSubClient after subscriber stop trigger
    Convert data in args["task_queue"]

    :param sub: current subscriber
    """
    for ts, msg in sub["args"]['task_queue'].items():
        sub["args"]["dataframe"][ts] = _msg_to_dataframe(msg)


def _msg_to_dataframe(msg):
    """
    Convert ros message to pandas DataFrame

    :param msg: ros messgae
    :return: DataFrame
    """
    pass
