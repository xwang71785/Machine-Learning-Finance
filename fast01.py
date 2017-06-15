# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 19:41:03 2017
Fast.ai
@author: 榴莲男爵
"""

import math, sys, os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets

np.set_printoptions(precision=4, linewidth=100)

def lin(a, b, x): return a * x + b
a = 3.
b = 8.
n = 30
x = np.random.random(n)
y = lin(a, b, x)

plt.scatter(x, y)

path = 'data/'
batch_size = 8

'''
from vgg16 import Vgg16

vgg = Vgg16() 

batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)
vgg.model.save_weights(path+'result/ft1.h5')
# reduce the learning rate to improve the performance
vgg.model.optimizer.lr = 0.01
vgg.fit(batches, val_batches, nb_epoch=1)
vgg.model.save_weights(path+'result/ft2.h5')
'''




