#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   utils.py    
@Contact    :   wyzlshx@foxmail.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/29 10:03    wxy        1.0         dataloader utils
"""

# import lib
from multiprocessing.dummy import Process
import os
import shutil
import signal
import time
import ctypes
import multiprocessing
import rospy


class MultiSubClient:
    """
    Ros multi subscriber client manager
    """

    def __init__(self, node_name='vrpn_listener'):
        """
        Init client obj, init ros node
        """
        rospy.init_node(node_name)

        self.content_init()
        # init interrupt operation
        self.worker_init()

    def content_init(self):
        """
        Init contents
        
        self.subscribers item structure: {name: {name, topic_type, sub_type, callback, args, subscriber}}
        name: sub id
        topic_type: for ROS use only
        sub_type: class of sub obj
        callback: for ROS use only
        args: pass to sub obj __init__
        subscriber: sub obj
        """
        self.subscribers = {}

        # Set by self.stop_all_subs, for multiprocess use
        self.global_unreg_flag = multiprocessing.Value(ctypes.c_bool, False)


    def worker_init(self):
        """
        Init worker signal callback, enable interrupt only in main process

        :return:
        """
        # SIGINT callback
        def sig_int(signal_num, frame):
            self.stop_all_subs()

        # change default singal callback
        signal.signal(signal.SIGINT, sig_int)

    def update_args(self, name, args):
        """
        Update client params, which will be passed to callbacks as args
        :name: subscriber name
        :args: args dict
        :return: dict
        """
        if name in self.subscribers.keys():
            self.subscribers[name]["args"].update(args)
        return self.subscribers[name]

    def add_sub(self, name: str, topic_type=object, callback=None, **kwargs) -> dict:
        """
        Add subscriber.

        :param name: sub topic name, sub identifier
        :param topic_type:
        :param callback: callback function, should accept 2 params (data, args)
        :return: self.subscribers, dict
        """
        self.subscribers[name] = dict(
            name=name, topic_type=topic_type, callback=callback, args={},
            subscriber=None, sub_type=kwargs.get("sub_type", rospy.Subscriber)
        )
        self.update_args(name, kwargs)

        # Pass gloabl unregister flag to callback_args
        self.update_args(name, dict(global_unreg_flag=self.global_unreg_flag))

        print_log("[DataLoader] task {} added.".format(name), log_obj=self.subscribers[name]["args"]["log_obj"], always_console=True)
        return self.subscribers

    def start_sub(self, name: str) -> dict:
        """
        Init and start a subscriber.
        :param name: sub topic name, sub identifier
        :return: current sub
        """
        if name in self.subscribers.keys():
            sub = self.subscribers[name]
            # subscribe topic
            if sub["subscriber"] is None:
                # init args
                if "before_start" in sub["args"].keys() and sub["args"]["before_start"] is not None:
                    sub["args"]["before_start"](sub)
                sub["subscriber"] = sub["sub_type"](name, sub["topic_type"], callback=sub["callback"], callback_args=sub["args"])
                print_log("[DataLoader] task {} started.".format(name), log_obj=self.subscribers[name]["args"]["log_obj"], always_console=True)
            else:
                print_log("[DataLoader] task {} already started.".format(name), log_obj=self.subscribers[name]["args"]["log_obj"], always_console=True)
        return sub

    def stop_sub(self, name: str) -> dict:
        """
        Stop a subscriber.
        :param name: sub topic name, sub identifier
        :return: current sub
        """
        if name in self.subscribers.keys() and self.subscribers[name]["subscriber"] is not None:
            sub = self.subscribers[name]
            sub["subscriber"].unregister()
            print_log("[DataLoader] task {} stopped.".format(name), log_obj=self.subscribers[name]["args"]["log_obj"])
            if "after_stop" in sub["args"].keys() and sub["args"]["after_stop"] is not None:
                    sub["args"]["after_stop"](sub)

    def get_running_tasks(self) -> list:
        running_list = []
        for name, sub in self.subscribers.items():
            if sub["subscriber"] is None:
                continue
            if sub["subscriber"].release_flag.value:
                continue
            running_list.append(name)
        return running_list

    def start_all_subs(self):
        """
        Start all subscribers.
        Use custom loop to avoid block.
        Use rospy.Subscriber.unregister() to stop subscription.
        Use rospy.wait_for_message() for single message.

        :return: None
        """
        self.global_unreg_flag.value = False
        for name, sub in self.subscribers.items():
            self.start_sub(name)

    def stop_all_subs(self):
        """
        Stop all subscribers.

        :return: None
        """
        self.global_unreg_flag.value = True
        for name, sub in self.subscribers.items():
            self.stop_sub(name)

    def print_info(self, stdscr):
        """
        Print current running infomation with curses

        :param stdscr: curses._CursesWindow 
        :return: None
        """
        stdscr.clear()
        stdscr.addstr(0, 0, "{}".format(ymdhms_time()))
        stdscr.addstr(1, 0, "Tasks:")
        stdscr.addstr(2, 0, "; ".join(self.get_running_tasks()))
        stdscr.addstr(3, 0, "Status:")
        idx = 4
        for sub in self.subscribers.values():
            if sub["subscriber"] is None:
                sub_info = "[{}] is not started or already stopped.\n".format(sub["name"])
            elif not sub["args"].get("info", None):
                sub_info = "[{}] does not support print-info.\n".format(sub["name"])
            else:
                sub_info = ("[{}]" + sub["args"]["info"]["formatter"]).format(sub["name"], *sub["args"]["info"]["data"])
            stdscr.addstr(idx, 0, sub_info)
            idx += 1
        stdscr.refresh()
        return


def clean_dir(dir):
    """
    Remove EVERYTHING under dir

    :param dir: target directory
    """
    if os.path.exists(dir):
        shutil.rmtree(dir, ignore_errors=True)

    os.makedirs(dir)


def ymdhms_time() -> str:
    """
    Return current time str for print use

    :return: time str
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def print_log(content, log_obj=None, always_console=False):
    """
    Custom print function.
    If log_obj is None, print to console as normal print;
    Else print to log_obj, use flush to refresh;
    If always_console, always print to console.

    :param content: print content str
    :param log_obj: log file object
    :param always_console: print to console even if already logged to log file
    """
    if log_obj is None:
        print("{} : {}".format(ymdhms_time(), content))
    else:
        print("{} : {}".format(ymdhms_time(), content), file=log_obj)
    if log_obj and always_console:
        print("{} : {}".format(ymdhms_time(), content))


def file_paths_from_dir(path, extension='.png', enable_print=True) -> list:
    """
    return specific files of the path and its sub-paths
    """
    if enable_print:
        print('Loading {} from {}'.format(extension, path))
    dir = os.listdir(path)
    filepaths = []
    for f in dir:
      if os.path.isdir(os.path.join(path, f)):
        file_paths_from_dir(os.path.join(path, f))

      if os.path.splitext(f)[1] == extension:
        filepaths.append(os.path.join(path, f))
    return filepaths


def filename_decoder(file_path) -> dict:
    dir, _ = os.path.split(file_path)
    filename, extension = os.path.splitext(_)
    params = [param.split('=') for param in filename.split('_')]
    params.append(["filepath", os.path.abspath(file_path)])
    return dict(params)


class PrintableValue:
    """
    multiprocessing.Value wrapper for better print.
    """
    def __init__(self, typecode_or_type, *args) -> None:
        self._value = multiprocessing.Value(typecode_or_type, *args)
    
    def value(self, v=None):
        if v is not None:
            self._value.value = v
        return self._value.value

    def __str__(self) -> str:
        return str(self.value())

    def __repr__(self) -> str:
        return str(self)
