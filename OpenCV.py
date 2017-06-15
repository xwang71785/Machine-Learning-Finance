# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:50:48 2016
Topic: Open CV
@author: wangx3
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as flt
import skimage as sim


picture = 'data/london.jpg'
# 读取图像文件，生成一个三维NumPy数组
img = cv2.imread(picture)
# 数组元素的格式是uint8
print (img.dtype)
# 图像的尺寸
rows = img.shape[0]
cols = img.shape[1]
# resize 
res = cv2.resize(img, None, fx=0.5, fy=0.5)

# Open CV中的颜色通道顺序是BGR
blue = res[:, :, 0]
green = res[:, :, 1]
red = res[:, :, 2]
# 转换成灰度图像
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
# grayn = np.array(gray)
film = 255 - gray
equ = cv2.equalizeHist(gray)


# rotation need generate a Matrix first
matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
rot = cv2.warpAffine(img, matrix, (cols, rows))

# 高斯模糊(用高斯核做卷积)，设定二维高斯核的标准差
guass = np.zeros(img.shape)
for i in range(3):
    guass[:, :, i] = flt.gaussian_filter(img[:, :, i], 5)
guass = np.uint8(guass)    #卷积结果是float，要转换成uint8

# Sobel导数滤波器
imx = np.zeros(gray.shape)
flt.sobel(gray, 1, imx)

imy = np.zeros(gray.shape)
flt.sobel(gray, 0, imy)

mag = np.sqrt(imx**2 + imy**2)

plt.imshow(guass)
plt.show()

"""
# 设置视频捕获
cap = cv2.VideoCapture(0)
# 获取视频帧并显示
while True:
    ret, img = cap.read()
    cv2.imshow('Video Test', img)
    key = cv2.waitKey(10)
    if key == 27:
        break
"""    
# waitkey函数会处于等待状态，否则显示会死机
pic = gray
cv2.imshow('sample', pic)
cv2.waitKey()

# Using Matplotlib to show the image
# Matplotlib中的颜色通道顺序是RGB
# cv2.COLOR_BGR2RGB用来转换通道顺序
#pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
"""
pic = plt.imread(picture)
plt.imshow(pic)
plt.axis('off')
plt.figure()    # 新建一个图像
plt.hist(pic.flatten(), 128)    #二维数组转换成一维数组
plt.show()
"""
