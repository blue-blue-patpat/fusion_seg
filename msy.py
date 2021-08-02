import numpy as np






file="__test__/2021-08-01 12:21:26/kinect/master/skeleton/id=0_tm=3161255_st=1627791733.2535143.npy"
content=np.load(file)

print(content)

a=content[0,0,0]
print(a)

b=content[0,0,1]
print(b)

c=content[0,0,2]
print(c)

d=content[:,:,:1]-a
print(d)

e=content[:,:,1:2]-b
print(e)

f=content[:,:,2:3]-c
print(f)

content[:,:,:1]=d
content[:,:,1:2]=e
content[:,:,2:3]=f

print(content)