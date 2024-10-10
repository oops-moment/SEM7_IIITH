import numpy as np
import math
a = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(a.shape)

print(a)

print(a[1][1])
a[1][0]=9 #-->mutable
# print(a)

print(a[:2]) # doubt

# 
b=a
print(b)
b[0][0]=90

print(a)  # more like a view , referencing to the same array
print(b) #why did changes propagate to a????


# multiple nested arrays 

a=np.array([[1,2,3],[4,5,6]])
print(a)
print(a[1][2])
print(a[1,2])

print(a.ndim)
print(a.shape)

print(a.size)
print(math.prod(a.shape))

print(a.dtype)

print(np.zeros((2,3)))

print(np.arange(2,9,2))

print(np.linspace(0,10,6))

# you can exp;ciotu write dtype here 
print(np.ones(2,dtype=np.int64))

arr= np.array([2,4,6,7,8,11,10])

print(np.sort(arr))
print(arr)

frist_arrsy=np.array([1,2,3,4])
sec0pd_arry = np.array([5,6,7,8])

print(np.concatenate((frist_arrsy,sec0pd_arry)))


x= np.array([[1,2],[3,4]])
y=np.array([[5,6]])

print(np.concatenate((x,y),axis=0))
# print(np.concatenate((x,y),axis=1))

# 2*2
# 1*2

#  axis confusion ? guce one more exampple for axis 1 for conceptual clarity

a= np.linspace(1,10,3)

print(a)

print(a.reshape(1,3))

data = np.array([1, 2, 3])

print(data[1])
print(data[0:2])
print(data[1:])
print(data[-2:])