import numpy as np
import math

a = np.array([[0,0,0],[1,2,3],[2,4,6]])
b=np.dot(a, a)
print(a)
print(b)

c=a[1,:]-a[0,:]

d=[[1,0,0],[0,c[1]/math.sqrt(c[1]*c[1]+c[2]*c[2]),-c[2]/math.sqrt(c[2]*c[2]+c[1]*c[1])],[0,c[2]/math.sqrt(c[1]*c[1]+c[2]*c[2]),c[1]/math.sqrt(c[1]*c[1]+c[2]*c[2])]]
print(c)
print(d)
e=np.dot(c,d)
print(e)

f=[[e[1]/math.sqrt(e[0]*e[0]+e[1]*e[1]),e[0]/math.sqrt(e[0]*e[0]+e[1]*e[1]),0],[-e[0]/math.sqrt(e[0]*e[0]+e[1]*e[1]),e[1]/math.sqrt(e[0]*e[0]+e[1]*e[1]),0],[0,0,1]]
g=np.dot(e,f)
print(g)