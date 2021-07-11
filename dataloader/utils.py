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
import rospy
import os
import shutil


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

    def add_sub(self, name: str, topic_type, callback, **kwargs) -> dict:
        """
        Add subscriber
        :param name: sub topic name, sub identifier
        :param topic_type:
        :param callback: callback function, should accept 2 params (data, args)
        :return: self.subscribers, dict
        """
        self.subscribers[name] = dict(
            name=name, topic_type=topic_type, callback=callback, args={}, subscriber=None
        )
        self.update_args(name, kwargs)
        return self.subscribers

    def start_sub(self, name: str) -> dict:
        """
        Init and start a subscriber.
        :param name: sub topic name, sub identifier
        :return: current sub
        """
        if name in self.subscribers.keys():
            # init args
            if "before_start" in self.subscribers[name]["args"].keys() and self.subscribers[name]["args"]["before_start"] is not None:
                self.subscribers[name]["args"]["before_start"](self.subscribers[name])
            # subscribe topic
            self.subscribers[name]["subscriber"] = rospy.Subscriber(name, self.subscribers[name]["topic_type"],
                                                                    callback=self.subscribers[name]["callback"],
                                                                    callback_args=self.subscribers[name]["args"])
        return self.subscribers.get(name, None)

    def stop_sub(self, name: str) -> dict:
        """
        Stop a subscriber.
        :param name: sub topic name, sub identifier
        :return: current sub
        """
        if name in self.subscribers.keys() and self.subscribers[name]["subscriber"] is not None:
            self.subscribers[name]["subscriber"].unregister()
            if "after_stop" in self.subscribers[name]["args"].keys() and self.subscribers[name]["args"]["after_stop"] is not None:
                    self.subscribers[name]["args"]["after_stop"](self.subscribers[name])
        return self.subscribers.get(name, None)

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


def clean_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
