# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:35:21 2017
Fast.ai MNIST
@author: wangx3
"""

batch_size = 16

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras
from keras.preprocessing import image
import numpy as np
# 自定义onehot函数
def onehot(x): return to_categorical(x)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Keras2基于TensorFlow, 缺省情况下遵循'channel_last'的data_format.
# 添加的通道维在最后，索引由1改为3
X_test = np.expand_dims(X_test, 3)
X_train = np.expand_dims(X_train, 3)
print(X_test.shape)

y_train = onehot(y_train)
y_test = onehot(y_test)
print(y_train[:5])

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def norm_input(x): return (x - mean_px) / std_px

def get_lin_model():
    model = Sequential([
            # 把channel first 改为channel last, 元组(1,28,28)改为(28,28,1)
            Lambda(norm_input, input_shape=(28, 28, 1)),
            Flatten(),
            Dense(10, activation='softmax')])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_fc_model():
    model = Sequential([
            # 把channel first 改为channel last, 元组(1,28,28)改为(28,28,1)
            Lambda(norm_input, input_shape=(28, 28, 1)),
            Flatten(),
            Dense(512, activation='softmax'),
            Dense(10, activation='softmax')])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_cnn_model():
    model = Sequential([
            # 把channel first 改为channel last, 元组(1,28,28)改为(28,28,1)
            Lambda(norm_input, input_shape=(28, 28, 1)),
            # Convolution2D升级为Conv2D
            # 卷积核参数改为元组类型(3, 3)
            Conv2D(32, (3, 3), activation='relu'),
            # Batchnorm 因为卷积层的data_format是channels_last，axis应为-1
            BatchNormalization(axis=1),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(),
            BatchNormalization(axis=1),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(axis=1),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            BatchNormalization(),
            Dense(512, activation='softmax'),
            BatchNormalization(),
            # Dropout 50%
            Dropout(0.5),
            Dense(10, activation='softmax')])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data augmentation
gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=64)
test_batches = gen.flow(X_test, y_test, batch_size=64)
# 大写的N改为小写的n，nb_epoch改写为epochs, nb_val_samples改写为validation_steps
'''
lm = get_lin_model()
lm.fit_generator(batches, batches.n, epochs=1, validation_data=test_batches,
                 validation_steps=test_batches.n)

fc = get_fc_model()
fc.fit_generator(batches, batches.n, epochs=1, validation_data=test_batches,
                 validation_steps=test_batches.n)

cnn = get_cnn_model()
cnn.fit_generator(batches, batches.n, epochs=1, validation_data=test_batches,
                 validation_steps=test_batches.n)
'''
# Ensembling
def fit_model():
    model = get_cnn_model()
    model.fit_generator(batches, batches.n, epochs=1, validation_data=test_batches,
                 validation_steps=test_batches.n)
    model.optimizer.lr = 0.1
    model.fit_generator(batches, batches.n, epochs=4, validation_data=test_batches,
                 validation_steps=test_batches.n)
    model.optimizer.lr = 0.01
    model.fit_generator(batches, batches.n, epochs=12, validation_data=test_batches,
                 validation_steps=test_batches.n)
    model.optimizer.lr = 0.001
    model.fit_generator(batches, batches.n, epochs=12, validation_data=test_batches,
                 validation_steps=test_batches.n)
    return model

models = [fit_model() for i in range(6)]
model_path = 'data/mnist/models/'
for i, m in enumerate(models):
    m.save_weights(model_path+str(i)+'.pkl')
evals = np.array([m.evaluate(X_test, y_test, batch_size=256) for m in models])
all_preds = np.stack([m.predict(X_test, batch_size=256) for m in models])
avg_preds = all_preds.mean(axis=0)
keras.metrics.categorical_accuracy(y_test, avg_preds).eval()

