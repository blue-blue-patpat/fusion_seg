#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   nokov_loader.py    
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/7/11 10:38    wxy        1.0         RealSense dataloader
"""

import gc
# import lib
import os
from multiprocessing import Process, Manager
from time import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import rospy

import realsense.Frame_Collection as fc
from dataloader.utils import clean_dir


class RealSenseSubscriber(Process):
    def __init__(self, name="RealSenseSub", topic_type=None, callback=None, **kwargs) -> None:
        """
        RealsenseSubscriber init

        :param name: implements rospy.Subscriber.topic_name
        :param topic_type: Not used, implements rospy.Subscriber.topic_type
        :param callback: Not used, implements rospy.Subscriber.callback
        :param kwargs: implements rospy.Subscriber.kwargs
        """
        super().__init__(daemon=True)
        # Designating image saving paths
        self.save_path = self.args.get("save_path", "./__test__/realsense_output")
        self.name = name
        self.args = kwargs.get("callback_args", {})
        # unregister flag: True if main process decides to unregister
        self.unreg_flag = Manager().Value(bool, False)
        # release flag: True if sub process is ready to be released
        self.release_flag = Manager().Value(bool, False)

        # Set parameters
        frame_rate = 30
        resolution = (848, 480)
        rs.config().enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, frame_rate)
        rs.config().enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, frame_rate)

        # Create device objects
        self.device_manager = fc.DeviceManager(rs.context(), rs.config())

        self.start()

    def unregister(self) -> None:
        """
        Implements rospy.Subscriber.unregister()
        Stop current Process

        :return: None
        """
        # set unregister flag and wait for release flag
        self.unreg_flag.value = True
        while not self.release_flag.value:
            pass

        if self.is_alive():
            print("[{}] uses multiprocess, terminating: {}".format(self.name, time()))
            self.device_manager.disable_streams()
            self.terminate()

    def run(self) -> None:
        """
        Process run function

        :return:
        """
        # Designating image saving paths
        clean_dir(os.path.join(self.save_path, 'color'))
        clean_dir(os.path.join(self.save_path, 'depth'))
        clean_dir(os.path.join(self.save_path, 'vertex'))

        all_pipeline, serial_total = self.device_manager.enable_all_devices()
        print(all_pipeline)

        frame_list = []

        # Create align objects
        align = rs.align(rs.stream.color)

        # # Initialize variables
        frame_count = 0
        # Threshhold = 30

        # Start collecting
        try:
            # wait for unregister flag
            while not self.unreg_flag.value:

                # Collect frames from all cameras sequentially
                for camera in serial_total:
                    # Collect and align a frame
                    pipeline = all_pipeline[camera]
                    frames = pipeline.wait_for_frames()
                    frame = align.process(frames)

                    # keep frame, or it will be released after 15 frames
                    # WARNING: frame should be collected by GC manually
                    frame.keep()
                    ros_ts = rospy.get_time()

                    # save data because realtime point cloud compute can only process 10 frames per sec
                    frame_list.append(dict(frame=frame, ros_ts=ros_ts, frame_count=frame_count))

                    # Using numpy
                    # path_depth = path_frame + "Depth_Frames/" + str(frame_count) + "_" + str(time_stamp) + "cam_" + str(camera) + ".png"
                    # np.save(path.depth, img_depth)

                    frame_count += 1

                    print("[{}] {} frames caught, {} frames waiting: {}".format(self.name, frame_count, len(frame_list),
                                                                                ros_ts))
        except Exception as e:
            print(e)

        print("[{}] {} frames saving... : {}".format(self.name, len(frame_list), time()))
        for f in frame_list:
            frame, ros_ts, frame_count = f["frame"], f["ros_ts"], f["frame_count"]

            # Get timestamp
            ts = frame.get_timestamp()

            # Transform color and depth images into numpy arrays
            color = frame.get_color_frame()
            depth = frame.get_depth_frame()

            img_color = np.array(color.get_data())
            img_depth = np.array(depth.get_data())

            # compute point cloud
            pc = rs.pointcloud()
            pc.map_to(color)
            points = pc.calculate(depth)

            vtx = np.asanyarray(points.get_vertices())
            vtx_coord = np.asanyarray(points.get_texture_coordinates())
            vtx = np.reshape(vtx, (-1, 3))
            vtx_coord = np.reshape(vtx_coord, (-1, 2))

            # Saving images
            filename = "id={}_tm={}_rostm={}.png".format(frame_count, ts, ros_ts)
            path_color = os.path.join(self.save_path, "color", filename)
            path_depth = os.path.join(self.save_path, "depth", filename)
            path_vertex = os.path.join(self.save_path, "vertex", filename)

            cv.imwrite(path_color, img_color)
            cv.imwrite(path_depth, img_depth)
            np.savez(path_vertex, vtx=vtx, coord=vtx_coord)

            # release frame space
            del frame

        # call GC manually
        gc.collect()
        print("[{}] {} frames saved: {}".format(self.name, len(frame_list), time()))

        # set release flag, ready to be released
        self.release_flag.value = True
