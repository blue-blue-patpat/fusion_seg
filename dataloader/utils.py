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
import os
import psutil
import signal
import rospy
import shutil
import time


class TaskFinishException(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MultiSubClient:
    """
    Ros multi subscriber client manager
    """

    def __init__(self, node_name='vrpn_listener'):
        """
        Init client obj, init ros node
        """
        rospy.init_node(node_name)
        # {name: {args, topic_type, callback, args, subscriber}}
        self.subscribers = {}
        self.parent_id = os.getpid()
        self.worker_init()

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
        Add subscriber
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
        print("DataLoader task {} added: {}".format(name, time.time()))
        return self.subscribers

    def start_sub(self, name: str) -> dict:
        """
        Init and start a subscriber.
        :param name: sub topic name, sub identifier
        :return: current sub
        """
        if name in self.subscribers.keys():
            sub = self.subscribers[name]
            # init args
            if "before_start" in sub["args"].keys() and sub["args"]["before_start"] is not None:
                sub["args"]["before_start"](sub)
            # subscribe topic
            sub["subscriber"] = sub["sub_type"](name, sub["topic_type"], callback=sub["callback"], callback_args=sub["args"])
            print("DataLoader task {} started: {}".format(name, time.time()))
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
            print("DataLoader task {} stopped: {}".format(name, time.time()))
            if "after_stop" in sub["args"].keys() and sub["args"]["after_stop"] is not None:
                    sub["args"]["after_stop"](sub)
        return sub

    def start_all_subs(self, block=False):
        """
        Start all subscribers.
        Use custom loop to avoid block.
        Use rospy.Subscriber.unregister() to stop subscription.
        Use rospy.wait_for_message() for single message.
        :param block: spin if True, default False. WARNING: this will block the thread.
        :return: None
        """
        for name, sub in self.subscribers.items():
            self.start_sub(name)
        if block:
            rospy.spin()

    def stop_all_subs(self):
        """
        Stop all subscribers.
        :return: None
        """
        for name, sub in self.subscribers.items():
            self.stop_sub(name)

    def worker_init(self):
        def sig_int(signal_num, frame):
            print('signal: %s' % signal_num)
            self.stop_all_subs()
            raise TaskFinishException
            # parent = psutil.Process(self.parent_id)
            # for child in parent.children():
            #     if child.pid != os.getpid():
            #         print("killing child: %s" % child.pid)
            #         child.kill()
            # print("killing parent: %s" % parent_id)
            # parent.kill()
            # print("suicide: %s" % os.getpid())
            # psutil.Process(os.getpid()).kill()
        signal.signal(signal.SIGINT, sig_int)


def clean_dir(dir):
    """
    Remove EVERYTHING under dir

    :param dir: target directory
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
