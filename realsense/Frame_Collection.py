import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from realsense.realsense_device_manager import Device
import os


# Get serial number list of connected device
def enumerate_connected_devices(context):
    connect_device = []
    for device in context.devices:
        if device.get_info(rs.camera_info.name).lower() != "platform camera":
            connect_device.append(device.get_info(rs.camera_info.serial_number))
    return connect_device

# Define class DeviceManager
class DeviceManager:

    # Initialization
    def __init__(self, context, pipeline_configuration):
        assert isinstance(context, type(rs.context()))
        assert isinstance(pipeline_configuration, type(rs.config()))
        self._context = context
        self._available_devices = enumerate_connected_devices(context)
        self._enabled_devices = {}
        self._config = pipeline_configuration
        self._frame_counter = 0

    # Enable device with expected options
    def enable_device(self, device_serial, enable_ir_emitter):

        # Create pipeline object
        pipeline = rs.pipeline()
        self._config.enable_device(device_serial)
        pipeline_profile = pipeline.start(self._config)

        # Configure depth camera
        sensor_depth = pipeline_profile.get_device().first_depth_sensor()
        sensor_depth.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
        sensor_depth.set_option(rs.option.global_time_enabled, 1 if enable_ir_emitter else 0)

        # Configure color camera
        sensor_color = pipeline.get_active_profile().get_device().query_sensors()[1]
        sensor_color.set_option(rs.option.global_time_enabled, 1)

        # Append enabled devices to certain list
        self._enabled_devices[device_serial] = (Device(pipeline, pipeline_profile))

        return pipeline_profile

    # Enable all devices connected to the PC
    def enable_all_devices(self, enable_ir_emitter=True):

        # Show connected devices
        print(str(len(self._available_devices)) + " devices have been found")

        # Create profile dict and serial list
        pipeline_profile_total = {}
        serial_total = []
        for serial in self._available_devices:

            # Enable
            pipeline_profile_total[serial]=self.enable_device(serial, enable_ir_emitter)

            # Append serial to list
            serial_total.append(serial)
            print(serial,type(serial))

        # Show serial list
        print(serial_total)

        # Get all enabled pipeline
        all_pipeline = self.get_all_pipeline()

        return all_pipeline,serial_total

    # Get all enabled pipeline
    def get_all_pipeline(self):
        all_pipeline = {}
        for (serial, device) in self._enabled_devices.items():
            all_pipeline[serial] = device.pipeline
        return all_pipeline

    # Disable all streams
    def disable_streams(self):
        self._config.disable_all_streams()

if __name__ == '__main__':
    '''
    Frame collection main
    '''

    # Designating image saving paths
    path_frame = "Frames/"

    # Set parameters
    frame_rate = 30
    resolution = (848, 480)
    rs.config().enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, frame_rate)
    rs.config().enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, frame_rate)

    # Create device objects
    device_manager = DeviceManager(rs.context(), rs.config())
    all_pipeline, serial_total = device_manager.enable_all_devices()
    print(all_pipeline)

    # Create align objects
    align = rs.align(rs.stream.color)

    # Initialize variables
    frame_count = 0
    Threshhold = 30

    # Start collecting
    try:
        print("start")
        while frame_count < Threshhold:

            # Collect frames from all cameras sequentially
            for camera in serial_total:

                # Collect and align a frame
                pipeline = all_pipeline[camera]
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)

                # Get timestamp
                time_stamp = frames.get_timestamp()

                # Transform color and depth images into numpy arrays
                img_color = np.array(frames.get_color_frame().get_data())
                img_depth = np.array(frames.get_depth_frame().get_data())

                # Saving images
                path_color = path_frame + "Color_Frames/" + str(frame_count) + "_" + str(time_stamp) + "cam_" + str(camera) + ".png"
                path_depth = path_frame + "Depth_Frames/" + str(frame_count) + "_" + str(time_stamp) + "cam_" + str(camera) + ".png"

                # Using numpy
                # path_depth = path_frame + "Depth_Frames/" + str(frame_count) + "_" + str(time_stamp) + "cam_" + str(camera) + ".png"
                # np.save(path.depth, img_depth)

                cv.imwrite(path_color, img_color)
                cv.imwrite(path_depth, img_depth)
                print(img_depth)

            # Continue to loop
            frame_count = frame_count + 1

            # Show progress
            percentage = 100 * frame_count / Threshhold
            if percentage % 1 == 0:
                print("Collection progress: {}%, with maximum frames {} per camera".format(percentage, Threshhold))

    except KeyboardInterrupt:
        device_manager.disable_streams()

    finally:
        print("Finished")
