# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:58:42 2017
AN OpenCV
@author: wangx3
"""

import numpy as np
import cv2
import scipy.ndimage as ndi


#要使用Haar cascade实现，仅需要把库修改为lbpcascade_frontalface.xml
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

img = cv2.imread('girls.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 识别输入图片中的人脸对象.返回对象的矩形尺寸
# 函数原型detectMultiScale(gray, 1.2,3,CV_HAAR_SCALE_IMAGE,Size(30, 30))
# gray需要识别的图片
# 1.03：表示每次图像尺寸减小的比例
# 5：表示每一个目标至少要被检测到4次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
# CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的最小最大尺寸
# faces：表示检测到的人脸目标序列
faces = face_cascade.detectMultiScale(gray, 1.03, 5)
for (x,y,w,h) in faces:
    if w+h>200:#//针对这个图片画出最大的外框
        img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
def is_inside(o,i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy +ih

def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
img = cv2.imread('people.jpg')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
found, w = hog.detectMultiScale(img)

found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)
            
for person in found_filtered:
    draw_person(img, person)
    
cv2.imshow('people detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
kernel3 = np.array([[-1,-1,-1], 
                    [-1, 8,-1], 
                    [-1,-1,-1]])

kernel5 = np.array([[-1,-1,-1,-1,-1], 
                    [-1, 1, 2, 1,-1], 
                    [-1, 2, 4, 2,-1], 
                    [-1, 1, 2, 1,-1], 
                    [-1,-1,-1,-1,-1]])

img = cv.imread('london.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_k3 = ndi.convolve(gray, kernel3)
img_k5 = ndi.convolve(gray, kernel5)
blurred = cv.GaussianBlur(gray, (11, 11), 0)
g_hpf = gray - blurred
edges = cv.Canny(gray, 200, 300)

#ret, thresh = cv.threshold(gray, 127, 255, 0)
#image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, 
#                                             cv.CHAIN_APPROX_SIMPLE)
#color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# img = cv.drawContours(color, contours, -1, (0,255,0), 2)

#img = cv.pyrDown(img)

cv.imshow('london', edges)
cv.waitKey()
cv.destroyAllWindows()

# camera = cv.VideoCapture(0)

'''

