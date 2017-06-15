# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:50:07 2017

@author: wangx3
"""

import cv2
import numpy as np
import time

class CaptureManager(object):
    def __init__(self, capture, previewWindowManager = None, shouldMirrorPreview = False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        
        self._startTime = None
        self._framesElapsed = np.long(0)
        self._fpsEstimate = None
        
    @property
    def channel(self):
        return self._channel
    
    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None
            
    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame
    
    @property   
    def isWritingImage (self):
        return self._imageFilename is not None
  
    @property
    def isWritingVideo(self):
        return self._videoFilename is not None
    
    def enterFrame(self):
        """Capture the next frame, if any."""
        # But first, check that any previous frame was exited.
        assert not self._enteredFrame, \'previous enterFrame() had no matching exitFrame()'
        
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()    
    
    
    