# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:25:48 2017
GetKeys
@author: wangx3
"""

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,.'aps$/\\":
    keyList.append(char)
    
def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys