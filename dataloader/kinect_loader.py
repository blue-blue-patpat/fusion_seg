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

import cv2 as cv
import rospy
import gc
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

    # img = args["bridge"].imgmsg_to_cv(msg, msg.encoding)
    img = _imgmsg_to_cv(msg)
    # cv.imshow('rgb', img)

    save_path = args.get("save_path", "./__test__/kinect_output")
    filename = "cmr={}_id={}_rostm={}.png".format(
        args.get("img_type", "kinectdefault"),
        args.get("frame_count", 0),
        ts)
    cv.imwrite(os.path.join(save_path, filename), img)

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


def _imgmsg_to_cv(img_msg, desired_encoding="passthrough"):
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
from pyk4a import Config, PyK4A, FPS, ColorResolution,DepthMode, WiredSyncMode, ImageFormat, PyK4ARecord
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
        clean_dir(os.path.join(self.save_path, "pcls"))
        frame_list = []
        frame_count = 0
        self.device.start()
        self.start_tm = time.time()
        # self.pyk4a.bodyTracker_start()
        self.infodata[3].value(1)

        # self.queue_threshold = 100
        # self.manual_queue = []
        self.use_manual_queue = False
        
        # threading function
        def process(frame, save_path, filename, infodata):
            path_color = os.path.join(save_path, "color", "{}.png".format(filename))
            path_depth = os.path.join(save_path, "depth", "{}.png".format(filename))
            path_point = os.path.join(save_path, "pcls", filename)

            np.save(path_point, frame.transformed_depth_point_cloud)
            # np.save(path_color, frame.color)
            # np.save(path_depth, frame.depth)
            cv.imwrite(path_color, frame.color)
            cv.imwrite(path_depth, frame.depth)

            infodata[0].value(infodata[0].value() + 1)

            del frame

        # init threading pool
        pool = Pool()
        
        try:
            # wait for main program unreg flag
            while not self.global_unreg_flag.value:
                frame = self.device.get_capture()
                sys_tm = time.time()
                if np.any(frame.color) and np.any(frame.depth):
                    timestamp = frame.color_timestamp_usec/1000000
                    filename = "id={}_st={}_dt={}".format(frame_count, sys_tm, timestamp)
                    
                    # add task
                    # if self.use_manual_queue:
                    pool.apply_async(process, (frame, self.save_path, filename, self.infodata))
                    # else:
                    #     process(frame, self.save_path, filename, self.infodata)

                    frame_count += 1
                    self.infodata[1].value(frame_count)
                    print_log("[{}] {} frames captured.".format(self.name, frame_count), log_obj=self.log_obj)

                    # if frame_count%300==0:
                    #     gc.collect()

                    # update info
                    if frame_count%30==0:
                        if self.infodata[2].value() < 28:
                            self.use_manual_queue = True
                        if self.use_manual_queue and self.infodata[2].value() >= 28:
                            self.use_manual_queue = False

                        running_tm = time.time()-self.start_tm
                        m = int(np.floor(running_tm/60))
                        s = int(running_tm - m*60)

                        # update pannel info
                        self.infodata[2].value(round(frame_count/running_tm, 2))
                        self.infodata[4].value(m)
                        self.infodata[5].value(s)

                        # update vis info
                        if not self.disable_visualization.value:
                            cv.namedWindow("kinect_{}".format(self.device_id),0)
                            cv.resizeWindow("kinect_{}".format(self.device_id),400,300)
                            cv.moveWindow("kinect_{}".format(self.device_id),600 + self.device_id*400,0)
                            frame_to_show = cv.resize(frame.color[:,:,:3], (400, 300), interpolation=cv.INTER_LANCZOS4)
                            cv.imshow("kinect_{}".format(self.device_id), frame_to_show)
                            cv.waitKey(10)

        except Exception as e:
            print_log(e, log_obj=self.log_obj, always_console=True)

        self.device.close()
        print_log("[{}] Collection task stopped, processing {} frames...".format(self.name, frame_count), log_obj=self.log_obj)
        if self.log_obj:
            self.log_obj.flush()

        # for item in self.manual_queue:
        #     pool.apply_async(process, item)

        self.infodata[2].value(-1)
        self.infodata[3].value(2)
        
        # wait for pool tasks
        pool.close()
        pool.join()

        print_log("[{}] {} frames saved".format(self.name, len(frame_list)), log_obj=self.log_obj)

        # ready to be released
        self.infodata[3].value(0)
        self.release_flag.value = True

        cv.destroyAllWindows()

        # suicide
        os.kill(os.getpid(), signal.SIGTERM)


"""
Use kinect with SDK
MKV
"""


class KinectMKVSubscriber(Process):
    """
    Kinect Subscriber
    """
    def __init__(self, name="KinectMKVSub", topic_type=None, callback=None, callback_args={}) -> None:
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
        clean_dir(self.save_path)
        frame_list = []
        frame_count = 0
        self.device.start()
        self.task_tm = time.time()
        self.run_init_flag = False

        self.infodata[3].value(1)

        record = PyK4ARecord(device=self.device, config=self.config, path=os.path.join(self.save_path, "out.mkv"))
        record.create()
        
        try:
            # wait for main program unreg flag
            while not self.global_unreg_flag.value:
                frame = self.device.get_capture()
                if not (np.any(frame.color) and np.any(frame.depth)):
                    continue
                if not self.run_init_flag:
                    self.start_tm = time.time()
                    with open(os.path.join(self.save_path, "info.txt"), "w") as f:
                        f.write("starttm={}_tasktm={}".format(self.start_tm, self.task_tm))
                    self.run_init_flag = True
                
                record.write_capture(frame)

                self.infodata[1].value(self.infodata[1].value() + 1)
                frame_count += 1
                
                # update info
                if frame_count%30==0:
                    if self.infodata[2].value() < 28:
                        self.use_manual_queue = True
                    if self.use_manual_queue and self.infodata[2].value() >= 28:
                        self.use_manual_queue = False

                    running_tm = time.time()-self.start_tm
                    m = int(np.floor(running_tm/60))
                    s = int(running_tm - m*60)

                    # update pannel info
                    self.infodata[2].value(round(frame_count/running_tm, 2))
                    self.infodata[4].value(m)
                    self.infodata[5].value(s)
                    
                    # update vis info
                    if not self.disable_visualization.value:
                        cv.namedWindow("kinect_{}".format(self.device_id),0)
                        cv.resizeWindow("kinect_{}".format(self.device_id),400,300)
                        cv.moveWindow("kinect_{}".format(self.device_id),600 + self.device_id*400,0)
                        frame_to_show = cv.resize(frame.color[:,:,:3], (400, 300), interpolation=cv.INTER_LANCZOS4)
                        cv.imshow("kinect_{}".format(self.device_id), frame_to_show)
                        cv.waitKey(10)

        except Exception as e:
            print_log(e, log_obj=self.log_obj, always_console=True)
        
        record.flush()
        record.close()

        self.device.close()
        print_log("[{}] Collection task stopped, processing {} frames...".format(self.name, frame_count), log_obj=self.log_obj)
        if self.log_obj:
            self.log_obj.flush()

        self.infodata[2].value(-1)
        self.infodata[3].value(2)

        print_log("[{}] {} frames saved".format(self.name, len(frame_list)), log_obj=self.log_obj)

        # ready to be released
        self.infodata[3].value(0)
        self.release_flag.value = True

        cv.destroyAllWindows()

        # suicide
        os.kill(os.getpid(), signal.SIGTERM)


"""
Kinect with Skeleton
Only 10pfs
"""


import kinect.k4a._k4a as _k4a
from kinect.k4a.config import config
from kinect.k4a.pyKinectAzure import pyKinectAzure as pyK4ASkeleton

class KinectSkeletonSubscriber(Process):
    def __init__(self, name="KinectSub", topic_type=None, callback=None, callback_args={}) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.device_id = callback_args.get("device_id", 0)        
        self.config = callback_args.get("config", config())
        self.save_path = callback_args.get(
            "save_path", "./__test__/kinect_output")
        
        callback_args["info"] = dict(formatter="\tcount={}/{}; \tfps={}; \tstatus={}; \t{}:{}", data=[
            PrintableValue(ctypes.c_uint32, 0),
            PrintableValue(ctypes.c_uint32, 0),
            PrintableValue(ctypes.c_double, -1.),
            PrintableValue(ctypes.c_uint8, 0),
            PrintableValue(ctypes.c_uint8, 0),
            PrintableValue(ctypes.c_uint8, 0)
        ])
        self.infodata = callback_args["info"]["data"]

        self.log_obj = callback_args.get(
            "log_obj", None)
        self.disable_visualization = Value(ctypes.c_bool, callback_args.get(
            "disable_visualization", False))

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

    def get_skeleton(self, depth_image) -> list:
        """
        Detect human and return skeletons.
        
        :return: bodys
        """
        # Get body segmentation image
        body_image_color = self.pyks.bodyTracker_get_body_segmentation()
        depth_color_image = cv.convertScaleAbs (depth_image, alpha=0.05)  #alpha is fitted by visual comparison with Azure k4aviewer results 
        depth_color_image = cv.cvtColor(depth_color_image, cv.COLOR_GRAY2RGB)
        combined_image = cv.addWeighted(depth_color_image, 0.8, body_image_color, 0.2, 0)

        bodys = []
        # Draw the skeleton
        for body in self.pyks.body_tracker.bodiesNow:
            skeleton2D = self.pyks.bodyTracker_project_skeleton(body.skeleton)
            combined_image = self.pyks.body_tracker.draw2DSkeleton(skeleton2D, body.id, combined_image)
            skeleton = []
            for joint in body.skeleton.joints:
                # level = int(joint.confidence_level)
                position = [p for p in joint.position.v]
                orientation = [p for p in joint.orientation.v]
                skeleton.append(position + orientation)
            bodys.append(skeleton)

        # Overlay body segmentation on depth image
        cv.imshow(self.name, combined_image)
        cv.waitKey(1)
        return bodys
    
    def run(self) -> None:
        """
        Process main function
        """
        # Designating image saving paths
        clean_dir(os.path.join(self.save_path, "color"))
        clean_dir(os.path.join(self.save_path, "depth"))
        clean_dir(os.path.join(self.save_path, "skeleton"))
        clean_dir(os.path.join(self.save_path, "pcls"))
        # frame_list = []
        frame_count = 0
        self.infodata[3].value(1)
        
        # threading function
        def process(color_image, depth_image, bodys, frame_count, save_path, timestamp, sys_tm, infodata):
            filename = "id={}_st={}_dt={}".format(frame_count, sys_tm, timestamp)
            path_color = os.path.join(save_path, "color", filename+".png")
            path_depth = os.path.join(save_path, "depth", filename+".png")
            # path_pcls = os.path.join(save_path, "pcls", filename)
            path_skelton = os.path.join(save_path, "skeleton", filename)

            # save
            color_image = cv.imdecode(color_image, -1)
            cv.imwrite(path_color, color_image)
            cv.imwrite(path_depth, depth_image)
            # np.save(path_pcls, point_cloud)
            body_arr = np.asarray(bodys)
            if np.any(body_arr):
                np.save(path_skelton, body_arr)

            infodata[0].value(infodata[0].value() + 1)

        # init threading pool
        pool = Pool()
        # init pyk4a
        self.pyks = pyK4ASkeleton()
        # Open device
        self.pyks.device_open(self.device_id)
        # Start cameras 
        self.pyks.device_start_cameras(self.config)
        self.start_tm = time.time()
        # Initialize the body tracker
        self.pyks.bodyTracker_start()
        try:
            # wait for main program unreg flag
            while not self.global_unreg_flag.value:
                self.pyks.device_get_capture()
                sys_tm = time.time()

                color_image_handle = self.pyks.capture_get_color_image()
                depth_image_handle = self.pyks.capture_get_depth_image()
                # TODO: slow
                # point_cloud_handle = self.pyks.transform_depth_image_to_point_cloud(depth_image_handle)

                timestamp = self.pyks.image_get_timestamp(color_image_handle)/1000000
		        
                if color_image_handle and depth_image_handle:
                    # Perform body detection
                    # TODO: slow
                    self.pyks.bodyTracker_update()
                    # Read and convert the image data to numpy array:
                    color_image = self.pyks.image_convert_to_numpy(color_image_handle)
                    depth_image = self.pyks.image_convert_to_numpy(depth_image_handle)
                    # point_cloud = self.pyks.image_convert_to_numpy(point_cloud_handle)

                    # TODO: slow
                    bodys = self.get_skeleton(depth_image)

                    # Release the image
                    self.pyks.image_release(color_image_handle)
                    self.pyks.image_release(depth_image_handle)
                    self.pyks.image_release(self.pyks.body_tracker.segmented_body_img)

                    # add task
                    pool.apply_async(process, (color_image, depth_image, bodys, frame_count, self.save_path, timestamp, sys_tm, self.infodata))
                    frame_count += 1
                    self.infodata[1].value(frame_count)
                    print_log("[{}] {} frames captured.".format(self.name, frame_count), log_obj=self.log_obj)
                    

                    if frame_count%5==0:
                        running_tm = time.time()-self.start_tm
                        m = int(np.floor(running_tm/60))
                        s = int(running_tm - m*60)

                        # update pannel info
                        self.infodata[2].value(round(frame_count/running_tm, 2))
                        self.infodata[4].value(m)
                        self.infodata[5].value(s)
            self.pyks.capture_release()
            self.pyks.body_tracker.release_frame()

        except Exception as e:
            print_log(e, log_obj=self.log_obj, always_console=True)

        cv.destroyAllWindows()
        self.pyks.device_stop_cameras()
        self.pyks.device_close()

        print_log("[{}] Collection task stopped, processing {} frames...".format(self.name, frame_count), log_obj=self.log_obj)
        if self.log_obj:
            self.log_obj.flush()

        self.infodata[2].value(-1)
        self.infodata[3].value(2)
        
        # wait for pool tasks
        pool.close()
        pool.join()

        print_log("[{}] {} frames saved".format(self.name, frame_count), log_obj=self.log_obj)

        # ready to be released
        self.infodata[3].value(0)
        self.release_flag.value = True

        # suicide
        os.kill(os.getpid(), signal.SIGTERM)

def _get_device_ids() -> dict:
    """
    Get Kinect device id dict 
    """
    # k4a = pyK4ASkeleton()
    # k4a.device_close()
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
    elif type == "alone":
        return Config(
                    camera_fps=FPS.FPS_30,
                    color_resolution=ColorResolution.RES_1536P,
                    depth_mode=DepthMode.NFOV_UNBINNED,
                    wired_sync_mode=WiredSyncMode.STANDALONE,)
    elif type == "skeleton_mas":
        return config(
	    			color_resolution=_k4a.K4A_COLOR_RESOLUTION_1536P,
                    camera_fps=_k4a.K4A_FRAMES_PER_SECOND_30,
                    depth_mode=_k4a.K4A_DEPTH_MODE_NFOV_UNBINNED,
                    wired_sync_mode=_k4a.K4A_WIRED_SYNC_MODE_MASTER,)
    elif type == "skeleton_sub":
        return config(
                    color_resolution=_k4a.K4A_COLOR_RESOLUTION_1536P,
                    camera_fps=_k4a.K4A_FRAMES_PER_SECOND_30,
                    depth_mode=_k4a.K4A_DEPTH_MODE_NFOV_UNBINNED,
                    wired_sync_mode=_k4a.K4A_WIRED_SYNC_MODE_SUBORDINATE,)
    else:
        return config()

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
