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
from multiprocessing import Process, Manager
import gc
from time import time
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
        self.unreg_flag = Manager().Value(bool, False)
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
        self.unreg_flag.value = True
        while not self.release_flag.value:
            pass

        if self.is_alive():
            print("[{}] uses multiprocess, terminating: {}".format(self.name, time()))
            self.device_manager.disable_streams()
            self.terminate()

    def run(self) -> None:
        print(os.getpid())
        # Designating image saving paths
        frame_count = 0
        self.save_path = self.args.get("save_path", "./__test__/realsense_output")
        clean_dir(os.path.join(self.save_path, 'color'))
        clean_dir(os.path.join(self.save_path, 'depth'))
        clean_dir(os.path.join(self.save_path, 'vertice'))

        all_pipeline, serial_total = self.device_manager.enable_all_devices()
        print(all_pipeline)

        frame_list = []

        # Create align objects
        align = rs.align(rs.stream.color)

        # # Initialize variables
        # frame_count = 0
        # Threshhold = 30

        # Start collecting
        print("start")
        try:
            while not self.unreg_flag.value:

                # Collect frames from all cameras sequentially
                for camera in serial_total:

                    # Collect and align a frame
                    pipeline = all_pipeline[camera]
                    frames = pipeline.wait_for_frames()
                    frame = align.process(frames)
                    frame.keep()
                    ros_ts = rospy.get_time()
                    frame_list.append(dict(frame=frame, ros_ts=ros_ts, frame_count=frame_count))

                    # Using numpy
                    # path_depth = path_frame + "Depth_Frames/" + str(frame_count) + "_" + str(time_stamp) + "cam_" + str(camera) + ".png"
                    # np.save(path.depth, img_depth)

                    frame_count += 1

                    print("[{}] {} frames caught, {} frames waiting: {}".format(self.name, frame_count, len(frame_list), ros_ts))
        except Exception as e:
            print(e)

        print("[{}] {} frames saving... : {}".format(self.name, len(frame_list), time()))
        for f in frame_list:
            frame,  ros_ts, frame_count= f["frame"], f["ros_ts"], f["frame_count"]

            # Get timestamp
            ts = frame.get_timestamp()

            # Transform color and depth images into numpy arrays
            color = frame.get_color_frame()
            depth = frame.get_depth_frame()

            img_color = np.array(color.get_data())
            img_depth = np.array(depth.get_data())

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
            path_vertice = os.path.join(self.save_path, "vertice", filename)
            
            cv.imwrite(path_color, img_color)
            cv.imwrite(path_depth, img_depth)
            np.savez(path_vertice, vtx=vtx, coord=vtx_coord)

            del frame

        gc.collect()
        print("[{}] {} frames saved: {}".format(self.name, len(frame_list), time()))
        self.release_flag.value = True
