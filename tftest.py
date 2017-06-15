# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:05:10 2017

@author: admin
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as inputdata
mnist = inputdata.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    result = sess.run([product])
    print (result)



