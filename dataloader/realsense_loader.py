#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   nokov_loader.py    
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/7/11 10:38    wxy        1.0         RealSense dataloader
"""

# import lib
import os
import signal
from multiprocessing import Process
from time import sleep
import pyrealsense2 as rs
import realsense.Frame_Collection as fc
from dataloader.utils import clean_dir
import numpy as np
import cv2 as cv
import rospy


class RealSenseSubscriber(Process):
    def __init__(self, name="RealSenseSub", topic_type=None, callback=None, **kwargs) -> None:
        super().__init__(daemon=True)
        # Designating image saving paths
        # self.path_frame = "Frames/"
        self.name = name
        self.args = kwargs.get("callback_args", {})

        # Set parameters
        frame_rate = 30
        resolution = (848, 480)
        rs.config().enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, frame_rate)
        rs.config().enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, frame_rate)

        # Create device objects
        self.device_manager = fc.DeviceManager(rs.context(), rs.config())

        self.start()

    def unregister(self) -> None:
        if self.is_alive():
            self.device_manager.disable_streams()
            self.terminate()

    def run(self) -> None:
        # Designating image saving paths
        frame_count = 0
        save_path = self.args.get("save_path", "./__test__/realsense_output")
        clean_dir(os.path.join(save_path, 'color'))
        clean_dir(os.path.join(save_path, 'depth'))

        all_pipeline, serial_total = self.device_manager.enable_all_devices()
        print(all_pipeline)

        # Create align objects
        align = rs.align(rs.stream.color)

        # # Initialize variables
        # frame_count = 0
        # Threshhold = 30

        # Start collecting
        print("start")
        try:
            while True:

                # Collect frames from all cameras sequentially
                for camera in serial_total:

                    # Collect and align a frame
                    pipeline = all_pipeline[camera]
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)

                    # Get timestamp
                    time_stamp = frames.get_timestamp()
                    ros_ts = rospy.get_time()

                    # Transform color and depth images into numpy arrays
                    img_color = np.array(frames.get_color_frame().get_data())
                    img_depth = np.array(frames.get_depth_frame().get_data())

                    # Saving images
                    filename = "cmr={}_id={}_tm={}_rostm={}.png".format(camera, frame_count, time_stamp, ros_ts)
                    path_color = os.path.join(save_path, "color", filename)
                    path_depth = os.path.join(save_path, "depth", filename)

                    # Using numpy
                    # path_depth = path_frame + "Depth_Frames/" + str(frame_count) + "_" + str(time_stamp) + "cam_" + str(camera) + ".png"
                    # np.save(path.depth, img_depth)

                    cv.imwrite(path_color, img_color)
                    cv.imwrite(path_depth, img_depth)
                    print(path_depth)
                    print(img_depth)

                    frame_count += 1
        except:
            pass
