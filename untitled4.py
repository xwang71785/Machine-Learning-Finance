# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:14:47 2017
Fast.ai Style Transfer
@author: wangx3
"""


import importlib
import utils2; importlib.reload(utils2)
from utils2 import *    # utils2 is developed by Jeremy Howard

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics

from vgg16_avg import VGG16_Avg

# Tell Tensorflow to use no more GPU RAM than necessary, it is defined in utils2
limit_mem()

path = '/data/datasets/imagenet/sample/'
dpath = '/data/jhoward/fast/imagenet/sample/'

fnames = glob.glob(path+'**/*.JPEG', recursive=True)
n = len(fnames); n

fn = fnames[50]; fn

img=Image.open(fnames[50]); img

rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]

deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)

img_arr = preproc(np.expand_dims(np.array(img), 0))
shp = img_arr.shape
























