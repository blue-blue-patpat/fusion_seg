#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   kinect_loader.py    
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/7/13 13:13    wxy        1.0         Azure Kinect dataloader
"""

# import lib
import os

import cv2
import rospy
from cv_bridge import CvBridge

from dataloader.utils import clean_dir


def kinect_loader_before_start(sub: dict):
    """
    MultiSubClient before subscriber start trigger
    Init args

    :param sub: current subscriber
    """
    sub["args"].update(dict(name=sub["name"], bridge=CvBridge(), imgs={}, frame_count=0))
    clean_dir(sub["args"].get("save_path", "./__test__/kinect_output"))


def kinect_loader_callback(msg, args):
    """
    Kinect ros data to dataframe

    :param msg: ros message
    :param args: subscriber["args"]
    :return: None
    """
    ts = rospy.get_time()

    # img = args["bridge"].imgmsg_to_cv2(msg, msg.encoding)
    img = _imgmsg_to_cv2(msg)
    # cv2.imshow('rgb', img)

    save_path = args.get("save_path", "./__test__/kinect_output")
    filename = "cmr={}_id={}_rostm={}.png".format(
        args.get("img_type", "kinectdefault"),
        args.get("frame_count", 0),
        ts)
    cv2.imwrite(os.path.join(save_path, filename), img)

    args["frame_count"] += 1
    print("[{}] {} frames saved, {} frames waiting: {}".format(
        args['name'], args["frame_count"], len(args['imgs']), ts))


def _encoding_to_dtype_with_channels(encoding):
    """
    Image encoding mapper, to avoid using cv_bridge

    :param encoding: img encoding str
    :return: numpy datatype, number of channels
    """
    if encoding == "bgra8":
        encoding = "8UC4"
    if "U" in encoding:
        dtype = "uint"
    elif "S" in encoding:
        dtype = "int"
    elif "F" in encoding:
        dtype = "float"
    else:
        dtype = ""
    dtype += encoding[:-3]

    channel = int(encoding[-1:])
    return dtype, channel


def _imgmsg_to_cv2(img_msg, desired_encoding="passthrough"):
    """
    Convert a sensor_msgs::Image message to an OpenCV :cpp:type:`cv::Mat`.
    MOdified based on cv_bridge, only support passthrough mode

    :param img_msg:   A :cpp:type:`sensor_msgs::Image` message
    :param desired_encoding:  The encoding of the image data, one of the following strings:

        * ``"passthrough"``
        * one of the standard strings in sensor_msgs/image_encodings.h

    :rtype: :cpp:type:`cv::Mat`
    :raises CvBridgeError: when conversion is not possible.

    If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
    Otherwise desired_encoding must be one of the standard image encodings

    This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.

    If the image only has one channel, the shape has size 2 (width and height)
    """
    import sys
    import numpy as np
    dtype, n_channels = _encoding_to_dtype_with_channels(img_msg.encoding)
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    if n_channels == 1:
        im = np.ndarray(shape=(img_msg.height, img_msg.width),
                        dtype=dtype, buffer=img_msg.data)
    else:
        im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                        dtype=dtype, buffer=img_msg.data)
    print(im.sum())
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        im = im.byteswap().newbyteorder()

    if desired_encoding == "passthrough":
        return im

    # from cv_bridge.boost.cv_bridge_boost import cvtColor2

    # try:
    #     res = cvtColor2(im, img_msg.encoding, desired_encoding)
    # except RuntimeError as e:
    #     raise RuntimeError(e)

    # return res
