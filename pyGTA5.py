# -*- coding: utf-8 -*-
"""
Spyder Editor
Py play GTA5
This is a temporary script file.
"""

import numpy as np
from PIL.ImageGrab import grab
import cv2
import time
from directkeys import ReleaseKey, PressKey

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0, 255, 255], 3)
    except:
        pass
    
# region of interested
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_img):
    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # Edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
    # define area of interest parameters
    vertices = np.array([[10, 500],[10, 300],[300, 200],[500,200],[800, 300],[800, 500]])
    processed_img = roi(processed_img, [vertices])
    
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 5)
    draw_lines(processed_img, lines)
    return processed_img


def main():
    last_time = time.time()
    while(True):
        screen = np.array(grab(bbox=(0, 40, 800, 640)))    # the size of area to be captured
        new_screen = process_img(screen)
        
        print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()





















