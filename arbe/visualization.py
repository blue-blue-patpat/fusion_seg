import numpy as np
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sklearn.cluster import DBSCAN
import cv2
from mpl_toolkits.mplot3d import Axes3D
import os,time

bag_file = "/home/nesc525/chen/3DSVC/arbe/arbe.bag"
bag = rosbag.Bag(bag_file, "r")
info = bag.get_type_and_topic_info()
bag_data = bag.read_messages()
# plt.ion()
for topic,msg,t in bag_data:
    # print(data)
    gen = point_cloud2.read_points(msg)
    xs = []
    ys = []
    zs = []
    v = []
    r = []
    # print(type(gen))
    #x,y,z,rgb,range,方位角，俯仰角，多普勒，能量
    for p in gen:
        # print(p)
        xs.append(p[0])
        ys.append((p[1]))
        zs.append(p[2])

        # print (" x : %.3f  y: %.3f  z: %.3f" %(p[0],p[1],p[2]))
    pcd = np.array((xs,ys,zs),dtype=float)
    pcd = np.transpose(pcd)

    db = DBSCAN(eps=0.35, min_samples=25).fit(pcd)
    labels = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # print(n_clusters_)
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    start = time.time()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = pcd[class_member_mask & core_samples_mask]
        # print(range)
        plt.plot(xy[:, 0], xy[:, 1], xy[:, 2],'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)
        # xy = data[class_member_mask & ~core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], xy[:, 2],'o', markerfacecolor=tuple(col),
        #          markeredgecolor='k', markersize=6)
    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    ax.set_zlim(0, 5)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # print("1:",time.time() - start)
    plt.pause(0.00001)
    # plt.ioff()
    # print("2",time.time()-start)
    # plt.show()