import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

bag_file = "2021-06-17-10-25-29.bag"
bag = rosbag.Bag(bag_file, "r")
info = bag.get_type_and_topic_info()
bag_data = bag.read_messages()
time = []
for topic,msg,t in bag_data:
    n = []
    data = []
    # for i in msg.header:
    #     n.append(i)
    # print(n)
    # for i in msg.frameTarget48bitBuffer:
    #     data.append(i)

    # print(msg.header.frame_id)
    # print(msg.header.seq)
    print(msg.fields)
    for i in msg.data:
        data.append(i)
    # print(data)
    gen = point_cloud2.read_points(msg)
    print(type(gen))
    # print(gen[0], gen[1])
    for p in gen:
        print(p)
    print(t)
    print("--------------------------")
    # print(data)
