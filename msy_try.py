import numpy as np

a = np.array([[0,0,0],[1,2,3],[4,5,6]])
b=np.dot(a, a)
print(a)
print(b)

c=a[1,:]-a[0,:]
c=np.transpose(c)
print(c)