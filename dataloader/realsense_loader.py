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
import gc
import ctypes
from multiprocessing import Process, Value
from multiprocessing.dummy import Pool
import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

import realsense.Frame_Collection as fc
from dataloader.utils import clean_dir, print_log, PrintableValue


class RealSenseSubscriber(Process):
    def __init__(self, name="RealSenseSub", topic_type=None, callback=None, callback_args={}) -> None:
        """
        RealsenseSubscriber init

        :param name: implements rospy.Subscriber.topic_name
        :param topic_type: Not used, implements rospy.Subscriber.topic_type
        :param callback: Not used, implements rospy.Subscriber.callback
        :param callback_args: implements rospy.Subscriber.kwargs
        """
        super().__init__(daemon=True)
        # init args
        self.name = name
        self.args = callback_args
        self.save_path = self.args.get("save_path", "./__test__/realsense_output")

        # Set parameters
        frame_rate = self.args.get("frame_rate", 30)
        resolution = self.args.get("resolution", (1920, 1080))
        rs.config().enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, frame_rate)
        rs.config().enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, frame_rate)

        # Create device objects
        self.device_manager = fc.DeviceManager(rs.context(), rs.config())
        
        # output config
        self.log_obj = callback_args.get("log_obj", None)
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

        print_log("[{}] started.".format(self.name), log_obj=self.log_obj, always_console=True)
        self.start()

    def unregister(self) -> None:
        """
        Implements rospy.Subscriber.unregister()

        :return: None
        """
        pass

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
        print_log("[{}] Pipelines: {}.".format(self.name, all_pipeline), log_obj=self.log_obj)

        frame_list = []

        # Create align objects
        align = rs.align(rs.stream.color)

        # # Initialize variables
        frame_count = 0
        self.infodata[0].value(0)
        self.infodata[1].value(0)

        self.start_tm = time.time()

        self.infodata[3].value(1)

        # threading function
        def process(frame, frame_count, save_path, infodata):
            filename = "id={}_tm={}.png".format(frame_count, frame.get_timestamp())
            path_color = os.path.join(save_path, "color", filename)
            path_depth = os.path.join(save_path, "depth", filename)
            # path_vertex = os.path.join(save_path, "vertex", filename)

            # Transform color and depth images into numpy arrays
            color = frame.get_color_frame()
            depth = frame.get_depth_frame()

            img_color = np.array(color.get_data())
            img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
            img_depth = np.array(depth.get_data())

            # # compute point cloud
            # pc = rs.pointcloud()
            # pc.map_to(color)
            # points = pc.calculate(depth)

            # vtx = np.asanyarray(points.get_vertices())
            # vtx_coord = np.asanyarray(points.get_texture_coordinates())
            # vtx = np.reshape(vtx, (-1, 3))
            # vtx_coord = np.reshape(vtx_coord, (-1, 2))

            # Saving images
            cv.imwrite(path_color, img_color)
            cv.imwrite(path_depth, img_depth)
            # np.savez(path_vertex, vtx=vtx, coord=vtx_coord)

            infodata[0].value(infodata[0].value() + 1)

            # release frame space
            del frame

        # init threading pool
        pool = Pool()

        # Start collecting
        try:
            # wait for unregister flag
            while not self.global_unreg_flag.value:

                # Collect frames from all cameras sequentially
                for camera in serial_total:
                    # Collect and align a frame
                    pipeline = all_pipeline[camera]
                    frames = pipeline.wait_for_frames()
                    frame = align.process(frames)

                    # keep frame, or it will be released after 15 frames
                    # WARNING: frame should be collected by GC manually
                    frame.keep()

                    # save data using thread pool
                    pool.apply_async(process, (frame, frame_count, self.save_path, self.infodata))

                    # Using numpy
                    # path_depth = path_frame + "Depth_Frames/" + str(frame_count) + "_" + str(time_stamp) + "cam_" + str(camera) + ".png"
                    # np.save(path.depth, img_depth)

                    frame_count += 1
                    self.infodata[1].value(frame_count)

                    print_log("[{}] {} frames caught, {} frames waiting.".format(self.name, frame_count, len(frame_list)), log_obj=self.log_obj)
                    
                    # update info
                    if frame_count%10==0:
                        running_tm = time.time()-self.start_tm
                        m = int(np.floor(running_tm/60))
                        s = int(running_tm - m*60)

                        # update pannel info
                        self.infodata[2].value(round(frame_count/running_tm, 2))
                        self.infodata[4].value(m)
                        self.infodata[5].value(s)

                        # update vis info
                        if not self.disable_visualization.value:
                            cv.namedWindow('realsense',0)
                            cv.resizeWindow('realsense',400,300)
                            cv.moveWindow('realsense',200,0)
                            img_color = frame.get_color_frame().get_data()
                            img_color = cv.cvtColor(np.array(img_color), cv.COLOR_RGB2BGR)
                            frame_to_show=cv.resize(img_color[:,:,:3],(400,300),interpolation=cv.INTER_LANCZOS4)
                            cv.imshow("realsense", frame_to_show)
                            key = cv.waitKey(10)


        except Exception as e:
            print_log(str(e), log_obj=self.log_obj, always_console=True)

        # Save after collect 
        print_log("[{}] {} frames saving...".format(self.name, len(frame_list)), log_obj=self.log_obj)

        self.infodata[2].value(-1)
        self.infodata[3].value(2)

        pool.close()
        pool.join()

        # call GC manually
        gc.collect()
        print_log("[{}] {} frames saved.".format(self.name, self.infodata[0], log_obj=self.log_obj, always_console=True))

        # set release flag, ready to be released
        self.infodata[3].value(0)
        self.release_flag.value = True

        cv.destroyAllWindows()
        
        # suicide
        os.kill(os.getpid(), signal.SIGTERM)
