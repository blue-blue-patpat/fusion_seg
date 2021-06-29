import rosbag
import sys
from PIL import Image
import os
import numpy as np
import cv2


if __name__ == '__main__':
    bag_dir = 'D:/JLUOneDrive/OneDrive - zju.edu.cn/Code/Arbe/'
    img_dir = os.path.join(bag_dir, 'image2')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    bag_file = os.path.join('arbe-rosbag_2021-06-14-17-18-14.bag')
    bag = rosbag.Bag(bag_file)
    index = 0
    imgname = os.path.join(img_dir, '{:0>5d}.jpg')
    for topic, msg, t in bag.read_messages(topics='/cv_camera/image_raw/compressed'):
        header = msg.header
        header_seq = header.seq 
        stamp_sec = header.stamp.secs
        stamp_nsec = header.stamp.nsecs
        data = msg.data #bytes
        img = np.frombuffer(data, dtype=np.uint8) #转化为numpy数组
        img = img.reshape(msg.height, msg.width)
        cv2.imwrite(imgname.format(index), img) #保存为图片
        print('{:0>5d} {} {} {}'.format(index, header_seq, stamp_sec, stamp_nsec))
        index += 1