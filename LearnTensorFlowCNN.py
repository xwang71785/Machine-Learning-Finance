# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:31:10 2017
TensorFlow CNN
@author: wangx3
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

in_units = 784
h1_units = 300

def weight_variable(shape):
    # tf.truncated_normal()将Weight初始化为截断的正态分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):    # 2维卷积函数，strides卷积核移动的步长，Padding的输入输出同样尺寸
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):    # 最大池化函数
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, in_units])    # 接收Sample数据
y_ = tf.placeholder(tf.float32, [None, 10])    # 接收Sample的Label数据
x_image = tf.reshape(x, [-1, 28, 28, 1])    # -1代表样本数量不固定， 1是通道数

w_conv1 = weight_variable([5, 5, 1, 32])    # 卷积核尺寸，通道数量，卷积核数量
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)    # Dropout比率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

# tf.reduce_mean()是求均值，tf.reduce_sum()是求和
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(5000):
    batch = mnist.train.next_batch(50)    # 批次量为50
    if i%100 == 0:    # 每100次迭代，打印训练效果
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print('step %d, training accuracy %g'%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    

test_accuracy = accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
print('test accuracy %g'%test_accuracy)    






