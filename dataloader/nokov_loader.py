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
from tf.transformations import *

from ros_client import MultiSubClient

"""
Use NOKOV Seeker to save offline data.
"""


def to_dataframe_callback(data, args):
    """
    NOKOV vrpn data to dataframe

    :param data: vrpn data
    :param args:
    :return: None
    """
    # print(data, args)
    df = args.get('nokov_df', pd.DataFrame())
    # TODO: check df reference; check data structure
    df.loc[len(df)] = [data.header.frame_id, data.header.stamp, data.pose.position.x, data.pose.position.y,
                       data.pose.position.z]
    print("nokov frame{} saved.".format(data.header.frame_id))


class NOKOVLoader:
    """
    NOKOV online dataloader

    work flow:

    init -> init_dataframe -> init_client
    -> start_sub -> stop_sub -> get_dataframe
    """

    def __init__(self):
        self.name = ''
        self.client = None
        self.df = pd.DataFrame()

    def init_dataframe(self, columns=None) -> pd.DataFrame:
        """
        Init dataframe

        :param columns:
        :return:
        """
        if columns is None:
            columns = ["idx", "global_ts", "x", "y", "z"]
        self.df = pd.DataFrame(columns=columns)
        return self.df

    def init_client(self, msg_name: str, client=None, callback=to_dataframe_callback) -> MultiSubClient:
        """
        Init NOKOV dataloader

        :param msg_name: subscribe message name
        :param client: client object, init a new client if not assigned
        :param callback: callback function
        :return: client
        """
        if client is None:
            self.client = MultiSubClient()
        else:
            self.client = client
        self.name = msg_name
        self.client.add_sub(self.name, PoseStamped, callback)
        self.client.update_params({
            'nokov_df': self.df
        })
        return self.client

    def update_params(self, params: dict) -> dict:
        """
        Update client params

        :param params:
        :return: client params
        """
        return self.client.update_params(params)

    def start_sub(self, auto_clear=True) -> dict:
        """
        Start this subscriber

        :param auto_clear: clear df if True, default True
        :return: current subscriber with details
        """
        if auto_clear:
            self.df.drop(self.df.index, inplace=True)
        return self.client.start_sub(self.name)

    def stop_sub(self) -> dict:
        """
        Stop this subscriber

        :return: current subscriber with details
        """
        return self.client.stop_sub(self.name)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get dataframe result

        :return: dataframe
        """
        return self.df
