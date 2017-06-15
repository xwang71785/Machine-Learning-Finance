# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:38:57 2017
Numpy
@author: wangx3
"""

# 数组的概念是基于Numpy的

import numpy as np
'''
n = 15
x = np.arange(n) ** 2
y = np.arange(n) ** 3
c = x + y
print(c)

print(x.dtype)
print(x.shape)

m = np.array([np.arange(n), np.arange(n)])    #如果两个向量长度不一样，不能生成多维数组
print(m)
print(m.dtype)
print(m.shape)

# 用嵌套列表生成多维数组
ma = np.array([[1, 2], [3, 4]], dtype=np.uint64)
print(ma[1, 0])
print(ma.dtype)

# start:end:step(end excluded)
print(x[3 : 13 : 2])

mb = np.arange(24).reshape(2,3,4)
print(mb)
# 冒号是数组下标通配符
print(mb[:,2,:])
f = mb.flatten()
print(f)

t = m.transpose()
print(t)
# Stack函数的输入是Tuple
x = x.reshape(3, 5)
y = y.reshape(3, 5)
h = np.hstack((x, y))
v = np.vstack((x, y))
d = np.dstack((x, y))
print(h)
print(v)
d = d.astype('int64')

# 其余和金融分析有关的Numpy函数参考Pandas
# Pandas的diff比Numpy更好用
# 移动平均数的计算可以用Pandas的rolling.mean()
'''

# mat()创建矩阵时，如果输入是matrix或ndarray对象，则不会创建副本
a = np.mat('1 2 3; 4 5 6; 7 8 9')
t = a.T
i = a.I
















