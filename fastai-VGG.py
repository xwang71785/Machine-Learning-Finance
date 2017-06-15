# -*- coding: utf-8 -*-
"""
Spyder Editor
Fast.ai .VGG
This is a temporary script file.
"""

t_path = 'data/dogscats/train'
v_path = 'data/dogscats/valid'
batch_size = 4

from vgg16 import Vgg16
vgg = Vgg16()
batches = vgg.get_batches(t_path, batch_size = batch_size)
val_batches = vgg.get_batches(v_path, batch_size = batch_size)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)

#估计模型参数是基于'channels_first'的，但是Keras2.0是基于'channels_last'
#要设置相关函数的data_format和keras.json中image_data_format为'channels_first'
#vgg16.py中还要修改一些参数名称
