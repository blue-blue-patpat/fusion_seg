from multiprocessing import Manager, Process
import shutil
import numpy as np
import os
from time import time
import cv2 as cv
from pyk4a import *

try:
    from dataloader.utils import clean_dir
except:
    def clean_dir(dir):
        """
        Remove EVERYTHING under dir

        :param dir: target directory
        """
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)


class KinectSubscriber(Process):
    def __init__(self, name="KinectSub", topic_type=None, callback=None, callback_args={}) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.config = callback_args.get("config", Config())
        self.device_id = callback_args.get("device_id", 0)
        self.save_path = callback_args.get("save_path", "./__test__/kinect_output")

        # unregister flag: True if main process decides to unregister
        self.unreg_flag = Manager().Value(bool, False)
        # release flag: True if sub process is ready to be released
        self.release_flag = Manager().Value(bool, False)
        self.device = callback_args.get("device", PyK4A(config=self.config, device_id=self.device_id))
        print("{} start".format(self.device_id))
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
            # self.device.close()
            self.terminate()

    def run(self) -> None:
        clean_dir(os.path.join(self.save_path, "color"))
        clean_dir(os.path.join(self.save_path, "depth"))
        frame_list = []
        frame_count = 0
        self.device.start()
        try:
            while not self.unreg_flag.value:
                frame = self.device.get_capture()
                timestamp = frame.color_system_timestamp_nsec
                if np.any(frame.color) and np.any(frame.depth):
                    frame_list.append(dict(frame=frame, timestamp=timestamp, frame_count=frame_count))

                # color = frame.color
                # depth = frame.depth
                # filename = "id={}_tm={}.png".format(frame_count, timestamp)
                # path_color = os.path.join(self.save_path, "color", filename)
                # path_depth = os.path.join(self.save_path, "depth", filename)
                # cv.imwrite(path_color, color)
                # cv.imwrite(path_depth, depth)
                    frame_count += 1
                    print("[{} {}] {}frames capturing... : {}".format(self.name, self.device_id, frame_count, time()))

        except Exception as e:
            print(e)
            # self.device.close()
            # exit()
        self.device.close()

        for f in frame_list:
            frame, timestamp, frame_count = f["frame"], f["timestamp"], f["frame_count"]
            color = frame.color
            depth = frame.depth

            filename = "id={}_tm={}.png".format(frame_count, timestamp)
            path_color = os.path.join(self.save_path, "color", filename)
            path_depth = os.path.join(self.save_path, "depth", filename)

            cv.imwrite(path_color, color)
            cv.imwrite(path_depth, depth)

        print("[{}] {} frames saved: {}".format(self.name, len(frame_list), time()))

        # set release flag, ready to be released
        self.release_flag.value = True


def _get_device_ids():
    cnt = connected_device_count()
    if not cnt:
        print("No devices available")
        exit()
    id_dict = {}
    print(f"Available devices: {cnt}")
    for device_id in range(cnt):
        device = PyK4A(device_id=device_id)
        device.open()
        id_dict.update({device.serial:device_id})
        device.close()
    return id_dict


def _get_config(type="mas"):
    if type == "mas":
        return Config(color_resolution=ColorResolution.RES_3072P,
                    depth_mode=DepthMode.WFOV_UNBINNED,
                    wired_sync_mode=WiredSyncMode.MASTER,)
    elif type == "sub":
        return Config(color_resolution=ColorResolution.RES_3072P,
                    depth_mode=DepthMode.WFOV_UNBINNED,
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
    # try:
    #     while True:
    #         pass
    # except KeyboardInterrupt as e:
    #     print(e)
    #     device_sub2.unreg_flag.value = False
    #     device_sub1.unreg_flag.value = False
    #     device_master.unreg_flag.value = False

    #     device_sub2.unregister()
    #     device_sub1.unregister()
    #     device_master.unregister()

    device_sub1.join()
    device_sub2.join()
    device_master.join()

    # device = KinectSubscriber(config=Config(
    #     color_resolution=ColorResolution.RES_1080P,
    #     depth_mode=DepthMode.NFOV_UNBINNED,
    #     synchronized_images_only=True,
    #     wired_sync_mode=WiredSyncMode.SUBORDINATE), device_id=0)
    # device.run()
