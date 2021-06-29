#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   ros_client.py    
@Contact    :   wyzlshx@foxmail.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/29 10:03    wxy        1.0         A simple implementation for ros multi subscriber client
"""

# import lib
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
        self.subscribers = {}
        self.params = {}

    def update_params(self, params):
        """
        Update client params, which will be passed to callbacks as args

        :param params: param dict
        :return: params
        """
        self.params.update(params)
        return self.params

    def add_sub(self, name: str, topic_type, callback) -> dict:
        """
        Add subscriber

        :param name: sub topic name, sub identifier
        :param topic_type:
        :param callback: callback function, should accept 2 params (data, args)
        :return: self.subscribers, dict
        """
        self.subscribers[name] = dict(
            name=name, topic_type=topic_type, callback=callback, subscriber=None
        )
        return self.subscribers

    def start_sub(self, name: str) -> dict:
        """
        Init and start a subscriber.

        :param name: sub topic name, sub identifier
        :return: current sub
        """
        if name in self.subscribers.keys() and self.subscribers["name"]["subscriber"] is None:
            self.subscribers[name]["subscriber"] = rospy.Subscriber(name, self.subscribers[name]["topic_type"],
                                                                    callback=self.subscribers[name]["callback"],
                                                                    callback_args=self.params)
        return self.subscribers.get(name, None)

    def stop_sub(self, name: str) -> dict:
        """
        Stop a subscriber.

        :param name: sub topic name, sub identifier
        :return: current sub
        """
        if name in self.subscribers.keys() and self.subscribers["name"]["subscriber"] is not None:
            self.subscribers[name]["subscriber"].unregister()
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
            self.start_sub(name)
