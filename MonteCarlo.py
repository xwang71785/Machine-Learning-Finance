# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:34:00 2017
MonteCarlo Simulation
@author: wangx3
"""

import random

def rollDice():
    roll = random.randint(1, 100)
    return roll

x=0
while x < 10:
    print(rollDice())
    x += 1
    
    