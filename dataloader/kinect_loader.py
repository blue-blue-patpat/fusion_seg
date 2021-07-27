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
# from cv_bridge import CvBridge

from dataloader.utils import clean_dir


"""
Use Kinect with ROS
"""


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


"""
Use kinect with SDK
"""

import ctypes
import signal
from multiprocessing import Value, Process
from multiprocessing.dummy import Pool
import shutil
import numpy as np
import os
import time
import cv2 as cv
from pyk4a import Config, PyK4A, FPS, ColorResolution,DepthMode, WiredSyncMode
from pyk4a import  connected_device_count

try:
    from dataloader.utils import clean_dir, print_log, PrintableValue
except:
    def clean_dir(dir):
        """
        Remove EVERYTHING under dir

        :param dir: target directory
        """
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    print_log = print


class KinectSubscriber(Process):
    """
    Kinect Subscriber
    """
    def __init__(self, name="KinectSub", topic_type=None, callback=None, callback_args={}) -> None:
        """
        KinectSubscriber init

        :param name: implements rospy.Subscriber.topic_name
        :param topic_type: Not used, implements rospy.Subscriber.topic_type
        :param callback: Not used, implements rospy.Subscriber.callback
        :param callback_args: implements rospy.Subscriber.kwargs
        """
        super().__init__(daemon=True)
        self.name = name
        
        # device config
        self.config = callback_args.get("config", Config())
        self.device_id = callback_args.get("device_id", 0)
        self.device = callback_args.get("device", PyK4A(
            config=self.config, device_id=self.device_id))

        self.save_path = callback_args.get(
            "save_path", "./__test__/kinect_output")
        
        
        self.log_obj = callback_args.get(
            "log_obj", None)
        self.disable_visualization = Value(ctypes.c_bool, callback_args.get(
            "disable_visualization", False))

        callback_args["info"] = dict(formatter="\tcount={}/{}; \tfps={}; \tstatus={}; \t{}:{}", data=[
            PrintableValue(ctypes.c_uint32, 0),
            PrintableValue(ctypes.c_uint32, 0),
            PrintableValue(ctypes.c_double, -1.),
            PrintableValue(ctypes.c_uint8, 0),
            PrintableValue(ctypes.c_uint8, 0),
            PrintableValue(ctypes.c_uint8, 0)
        ])
        self.infodata = callback_args["info"]["data"]

        # init flag
        # unregister flag: True if main process decides to unregister
        self.global_unreg_flag = callback_args.get(
            "global_unreg_flag", Value(ctypes.c_bool, False))
        self.unreg_flag = Value(ctypes.c_bool, False)
        # release flag: True if sub process is ready to be released
        self.release_flag = Value(ctypes.c_bool, False)

        print_log("[{}] {} started.".format(self.name, self.device_id), log_obj=self.log_obj, always_console=True)

        # start process
        self.start()

    def unregister(self) -> None:
        """
        Implements rospy.Subscriber.unregister()

        :return: None
        """
        pass

    def run(self) -> None:
        """
        Process main function
        """
        # Designating image saving paths
        clean_dir(os.path.join(self.save_path, "color"))
        clean_dir(os.path.join(self.save_path, "depth"))
        frame_list = []
        frame_count = 0
        self.device.start()
        self.start_tm = time.time()
        self.infodata[3].value(1)
        
        # threading function
        def process(frame, frame_count, save_path, sys_tm, infodata):
            timestamp = frame.color_timestamp_usec
            filename = "id={}_tm={}_st={}.png".format(frame_count, timestamp, sys_tm)
            path_color = os.path.join(save_path, "color", filename)
            path_depth = os.path.join(save_path, "depth", filename)
            cv.imwrite(path_color, frame.color)
            cv.imwrite(path_depth, frame.depth)

            infodata[0].value(infodata[0].value() + 1)

        # init threading pool
        pool = Pool()
        
        try:
            # wait for main program unreg flag
            while not self.global_unreg_flag.value:
                frame = self.device.get_capture()
                
                sys_tm = time.time()
                if np.any(frame.color) and np.any(frame.depth):
                    
                    # add task
                    pool.apply_async(process, (frame, frame_count, self.save_path, sys_tm, self.infodata))

                    frame_count += 1
                    self.infodata[1].value(frame_count)
                    print_log("[{}] {} frames captured.".format(self.name, frame_count), log_obj=self.log_obj)

                    # update info
                    if frame_count%15==0:
                        running_tm = time.time()-self.start_tm
                        m = int(np.floor(running_tm/60))
                        s = int(running_tm - m*60)

                        # update pannel info
                        self.infodata[2].value(round(frame_count/running_tm, 2))
                        self.infodata[4].value(m)
                        self.infodata[5].value(s)

                        # update vis info
                        if not self.disable_visualization.value:
                            if self.device_id==0:
                                cv.namedWindow("kinect_{}".format(self.device_id),0)
                                cv.resizeWindow("kinect_{}".format(self.device_id),400,300)
                                cv.moveWindow("kinect_{}".format(self.device_id),600,0)
                                frame_to_show = cv.resize(frame.color[:,:,:3], (400, 300), interpolation=cv.INTER_LANCZOS4)
                                cv.imshow("kinect_{}".format(self.device_id), frame_to_show)
                                cv.waitKey(10)
                            elif self.device_id==1:
                                cv.namedWindow("kinect_{}".format(self.device_id),0)
                                cv.resizeWindow("kinect_{}".format(self.device_id),400,300)
                                cv.moveWindow("kinect_{}".format(self.device_id),1000,0)
                                frame_to_show = cv.resize(frame.color[:,:,:3], (400, 300), interpolation=cv.INTER_LANCZOS4)
                                cv.imshow("kinect_{}".format(self.device_id), frame_to_show)
                                cv.waitKey(10)
                            elif self.device_id==2:
                                cv.namedWindow("kinect_{}".format(self.device_id),0)
                                cv.resizeWindow("kinect_{}".format(self.device_id),400,300)
                                cv.moveWindow("kinect_{}".format(self.device_id),1400,0)
                                frame_to_show = cv.resize(frame.color[:,:,:3], (400, 300), interpolation=cv.INTER_LANCZOS4)
                                cv.imshow("kinect_{}".format(self.device_id), frame_to_show)      
                                cv.waitKey(10)

        except Exception as e:
            print_log(e, log_obj=self.log_obj, always_console=True)

        self.device.close()
        print_log("[{}] Collection task stopped, processing {} frames...".format(self.name, frame_count), log_obj=self.log_obj)
        self.log_obj.flush()

        self.infodata[2].value(-1)
        self.infodata[3].value(2)
        
        # wait for pool tasks
        pool.close()
        pool.join()

        print_log("[{}] {} frames saved".format(self.name, len(frame_list)), log_obj=self.log_obj)

        # ready to be released
        self.infodata[3].value(0)
        self.release_flag.value = True

        # suicide
        os.kill(os.getpid(), signal.SIGTERM)


def _get_device_ids() -> dict:
    """
    Get Kinect device id dict 
    """
    cnt = connected_device_count()
    if not cnt:
        print("No devices available")
        exit()
    id_dict = {}
    # print(f"Available devices: {cnt}")
    for device_id in range(cnt):
        device = PyK4A(device_id=device_id)
        if device.opened:
            device.close()
        device.open()
        id_dict.update({device.serial: device_id})
        device.close()
    return id_dict


def _get_config(type="mas") -> Config:
    """
    Get Kinect Config by character
    """
    if type == "mas":
        return Config(
                    camera_fps=FPS.FPS_30,
                    color_resolution=ColorResolution.RES_1536P,
                    depth_mode=DepthMode.NFOV_UNBINNED,
                    wired_sync_mode=WiredSyncMode.MASTER,)
    elif type == "sub":
        return Config(
                    camera_fps=FPS.FPS_30,
                    color_resolution=ColorResolution.RES_1536P,
                    depth_mode=DepthMode.NFOV_UNBINNED,
                    wired_sync_mode=WiredSyncMode.SUBORDINATE,)
    else:
        return Config()


if __name__ == "__main__":
    id_dict = _get_device_ids()
    print(id_dict)
    device_sub1 = KinectSubscriber("KinectSub1", callback_args=dict(config=_get_config("sub"),
                                                                    device_id=id_dict["000053612112"],
                                                                    save_path="./__test__/kinect_output/sub1"))
    device_sub2 = KinectSubscriber("KinectSub2", callback_args=dict(config=_get_config("sub"),
                                                                    device_id=id_dict["000176712112"],
                                                                    save_path="./__test__/kinect_output/sub2"))
    device_master = KinectSubscriber("KinectMaster", callback_args=dict(config=_get_config("mas"),
                                                                        device_id=id_dict["000326312112"],
                                                                        save_path="./__test__/kinect_output/master"))
    device_sub1.join()
    device_sub2.join()
    device_master.join()
