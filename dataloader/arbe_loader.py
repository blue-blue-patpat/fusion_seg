#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   arbe_loader.py
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/28 22:34    wxy        1.0         arbe dataloader
"""

# import lib
from __future__ import generators
from multiprocessing import Value
from threading import Timer
import ctypes
import time
import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool
import rospy
from sensor_msgs import point_cloud2
from dataloader.utils import clean_dir, print_log
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
class ArbeSubscriber(rospy.Subscriber):
    """
    Arbe Subscriber
    """
    def __init__(self, name, data_class, callback=None, callback_args=None,
                 queue_size=None, buff_size=65536, tcp_nodelay=False):
        # clear dir before start
        clean_dir(callback_args.get('save_path', './__test__/default/arbe/'))
    
        super().__init__(name, data_class, callback=callback, callback_args=callback_args, queue_size=queue_size, buff_size=buff_size, tcp_nodelay=tcp_nodelay)
        
        # True if self is ready to be released
        self.release_flag = Value(ctypes.c_bool, False)

        self.callback_args.update(dict(name=name, dataframe={}, task_queue={}, start_tm=time.time(),
            pool=Pool(),fig = plt.figure(1),
            info=dict(formatter="\tcount={}/{}; \tfps={}; \tstatus={}; \t{}:{}", data=[0, 0, -1, 1, 0, 0])))

    def unregister(self):
        """
        Stop task
        override Subscriber.unregister
        """
        # prevent super.unregister removing args
        args = self.callback_args
        super().unregister()
        self.callback_args = args
        #@plt.switch_backend("agg")
        plt.close("all")
        # wait until no more tasks will be added to pool
        self._close_pool()
        return

    def _close_pool(self):
        """
        Check and close pool using Timer.
        """
        # check waiting tasks
        waiting_count = self.callback_args["info"]["data"][1] - self.callback_args["info"]["data"][0]
        # if some tasks are still waiting, start a Timer and check later
        if waiting_count > 0:
            self.timer = Timer(3, self._close_pool)
            self.timer.start()
            return

        # close and wait until pool is terminated
        self.callback_args["pool"].close()
        self.callback_args["pool"].join()
        # ready to be released
        self.release_flag.value = True


def arbe_process(msg, frame_count, ts, save_path, infodata):
    # _msg_to_dataframe(msg).to_csv(os.path.join(save_path, '{}.csv'.format(ts)))
    np.save(os.path.join(save_path, 'id={}_ts={}'.format(frame_count, ts)), np.array(_msg_to_dataframe(msg)))
    infodata[0] += 1


def arbe_loader_callback(msg, args):
    """
    ros data stream to dataframe

    :param data: vrpn message
    :param args: dict(dataframe, name, task_queue)
    :return: None
    """
    ts = rospy.get_time()
    # callback may be triggered before __init__ completes. if pool is not started yet, ignore current frame
    if args.get("pool", None) is None:
        args["start_tm"] = time.time()
        return

    save_path = args.get('save_path', './__test__/default/arbe/')
    
    # if head is not recorded, save head to file
    if not args.get("headline", False):
        args["headline"] = True
        f = open(os.path.join(save_path, "headline.txt"), "w")
        f.write(",".join([item.name for item in msg.fields]))
        f.close()
    
    # add task
    args["pool"].apply_async(arbe_process, (msg, args["info"]["data"][1], ts, save_path, args["info"]["data"]))
    print_log("[{}] {} frames captured.".format(
        args['name'], args["info"]["data"][1]), args["log_obj"])

    #visualization
    #plt.switch_backend("tkagg")
    if args["info"]["data"][0]%15==0:
        _show_pc_img(msg, args["fig"])

    # update pannel info
    running_tm = time.time()-args["start_tm"]
    m = int(np.floor(running_tm/60))
    s = int(running_tm - m*60)
    fps = round((args["info"]["data"][1]) / running_tm, 2)
    args["info"]["data"][1] += 1
    args["info"]["data"][2] = fps
    args["info"]["data"][4] = m
    args["info"]["data"][5] = s

    if args.get('force_realtime', True):
        args["dataframe"][ts] = _msg_to_dataframe(msg)
    else:
        args["task_queue"][ts] = msg


# def arbe_loader_after_stop_abandoned(sub: dict):
#     for ts, msg in sub["args"]['msg_list'].items():
#         gen = point_cloud2.read_points(msg)
#         start_t = time.time()
#         last_t = start_t
#         for p in gen:
#             s = time.time()

#             row_data = [sub["args"]["frame_id"], ts]
#             row_data += list(p)

#             sub['args']['arr'].append(row_data)
#             d = time.time()
#             print(d-s, d - last_t)
#             last_t = d
#         end_t = time.time()
#         print("{} frame saved at {}, total {}, time period {}.".format(sub["args"]["frame_id"], ts, len(sub["args"]["arr"]), end_t - start_t))
#         sub["args"]["frame_id"] += 1
#     sub["args"]["dataframe"] = pd.DataFrame(sub['args']['arr'], columns=sub["args"]["column"])


def arbe_loader_after_stop(sub: dict):
    """
    MultiSubClient after subscriber stop trigger
    Convert data in args["task_queue"]

    :param sub: current subscriber
    """
    for ts, msg in sub["args"]['task_queue'].items():
        sub["args"]["dataframe"][ts] = _msg_to_dataframe(msg)
        

def _msg_to_dataframe(msg) -> pd.DataFrame:
    """
    Convert ros message to pandas DataFrame

    :param msg: ros messgae
    :return: DataFrame
    """
    return pd.DataFrame(_point_cloud_loader(msg), columns=[item.name for item in msg.fields])


def _point_cloud_loader(cloud):
    """
    Implementation of point_cloud2.read_points
    Return list directly, rather than Generator to speed up stream read.

    :param sub: current subscriber
    """
    import struct
    fmt = point_cloud2._get_struct_fmt(cloud.is_bigendian, cloud.fields, None)
    width, height, point_step, row_step, data = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data
    unpack_from = struct.Struct(fmt).unpack_from
    ret = []

    for v in range(height):
        offset = row_step * v
        for u in range(width):
            ret.append(unpack_from(data, offset))
            offset += point_step
    return ret


def _show_pc_img(msg, fig):
    frame = np.array(_point_cloud_loader(msg))

    plt.ion()

    pcd = frame[:,0:3]
    # db = DBSCAN(eps=0.35, min_samples=25).fit(pcd)
    # labels = db.labels_
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print(n_clusters_)

    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #         for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.clf()
    #mngr = plt.get_current_fig_manager()  # 获取当前figure manager
    #mngr.window.wm_geometry("+380+310")  # 调整窗口在屏幕上弹出的位置
    ax = fig.add_subplot(111, projection='3d')
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    col = (0, 0, 0, 1)

    #     class_member_mask = (labels == k)
    #     pcd_visible = pcd[class_member_mask & core_samples_mask]
    plt.plot(pcd[:, 0], pcd[:, 1], pcd[:, 2],'o', markerfacecolor=col,
            markeredgecolor='k', markersize=5)

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    ax.set_zlim(0, 10)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.pause(0.0001)
    #plt.switch_backend("agg")
    plt.ioff()
