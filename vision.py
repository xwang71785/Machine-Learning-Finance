# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:10:20 2017
Programming Computer Vision with Python
@author: wangx3
"""

import numpy as np
import PIL.Image as im
import matplotlib.pylab as plb
import scipy.ndimage.filters as fil
import cv2

path = 'data/'

image = im.open(path+'mountain.jpg')
data = plb.array(image)
gray = image.convert('L')
box = (100, 100, 400, 400)
regin = image.crop(box)
output = regin.rotate(45)
data2 = np.zeros(data.shape)
data3 = np.zeros(data.shape)
for i in range(3):
    data2[:,:,i] = fil.gaussian_filter(data[:,:,i], 5)
    data3[:,:,i] = fil.sobel(data[:,:,i], 1)
data2 = np.uint8(data3)
#data = plb.array(gray)
#plb.figure()
#plb.hist(data.flatten(), 128)
plb.imshow(data3)
# plb.title('Plotting: Mountain')
plb.show()

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyCamera')
print('Show camera feed. ESC to stop')
success, frame = cameraCapture.read()
while success:
    cv2.imshow('MyCamera', frame)
    success, frame = cameraCapture.read()
    key = cv2.waitKey(10)
    if key == 27:
        break
cv2.destroyWindow('MyCamera')
cameraCapture.release()


