import numpy as np
import math
import open3d as o3d


def ChangeMatrix(file):
    content=np.load(file)

    '''
    pcd=o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([pcd])
    #print(content)
    '''

    a=content[0,0]

    b=content[0,1]

    c=content[0,2]

    d=content[:,0:1]-a

    e=content[:,1:2]-b

    f=content[:,2:3]-c

    content[:,0:1]=d
    content[:,1:2]=e
    content[:,2:3]=f

    content=content[:,:3]
    #print(content)

    def cos(a,b):
        return a/math.sqrt(a*a+b*b)

    g=content[1,:]

    h=[[1, 0, 0],
        [0, cos(g[1],g[2]), -cos(g[2],g[1])],
            [0, cos(g[2],g[1]), cos(g[1],g[2])]]

    i=np.dot(g,h)

    j=[[cos(i[1], i[0]), cos(i[0],i[1]),0],
        [-cos(i[0],i[1]), cos(i[1],i[0]), 0],
            [0, 0, 1]]

    k=np.dot(h,j)
    l=np.dot(content,k)

    m=[[cos(l[18,0],l[18,2]),0,-cos(l[18,2],l[18,0])],
        [0,1,0],
            [cos(l[18,2],l[18,0]),0,cos(l[18,0],l[18,2])]]

    n=np.dot(l,m)
    print(n)
    print(n[22])

file="__test__/2021-07-31 21:35:50/arbe/id=0_ts=1627738558.816022.npy"
ChangeMatrix(file)