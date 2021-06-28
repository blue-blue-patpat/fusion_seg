#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import *
import numpy as np

rospy.init_node('vrpn_listener')
pub = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=10)
pub1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=10)

max_v, max_w = 0.22, 2.84
r, T = 0.75, 40
a, b = 0.25, 0.2
cnt = 0

#Leader Robot
w = 2 * np.pi / T
v = w * r
twist = Twist() 

#Follower
v1, w1 = 0., 0.
twist1 = Twist()

#Velocity Threshold
def max_vel(tmp_v, tmp_w):
    if tmp_v >= max_v:
        tmp_v = max_v
    if abs(tmp_w) >= max_w:
        tmp_w = np.sign(tmp_w) * max_w

    return tmp_v, tmp_w


while not rospy.is_shutdown():
    pos_0 = rospy.wait_for_message('/vrpn_client_node/tb3_0/pose', PoseStamped)
    pos_1 = rospy.wait_for_message('/vrpn_client_node/tb3_1/pose', PoseStamped)

    pos = [pos_0.pose.position.x / 1000., pos_0.pose.position.y/ 1000.]
    pos1 = [pos_1.pose.position.x/ 1000., pos_1.pose.position.y/ 1000.]
    # data format: x y z w
    pose = euler_from_quaternion((pos_0.pose.orientation.x, pos_0.pose.orientation.y,
                                  pos_0.pose.orientation.z, pos_0.pose.orientation.w))
    pose1 = euler_from_quaternion((pos_1.pose.orientation.x, pos_1.pose.orientation.y,
                                   pos_1.pose.orientation.z, pos_1.pose.orientation.w))
    yaw, yaw1 = pose[2], pose1[2]

    # algorithm
    if abs(abs(yaw) - np.pi) <= 0.1 and cnt >= 10:
        w *= -1
        cnt = 0

    dist = np.sqrt((pos[0] - pos1[0]) ** 2 + (pos[1] - pos1[1]) ** 2)
    v1 = a * dist
    w1 = b * np.sign(yaw - yaw1) * abs(np.arctan2(pos[1] - pos1[1], pos[0] - pos1[0]))
    print(dist, abs(yaw - yaw1))
    if dist <= 0.25:
        v1 = 0.
    if abs(yaw - yaw1) <= 0.1:
        w1 = 0.

    # publish
    v, w = max_vel(v, w)
    twist.linear.x = v
    twist.angular.z = w
    pub.publish(twist)

    v1, w1 = max_vel(v1, w1)
    twist1.linear.x = v1
    twist1.angular.z = w1
    pub1.publish(twist1)

    cnt += 1

    # print('----- pos0 -----')
    # print(pos)
    # print(yaw * 180 / np.pi)
