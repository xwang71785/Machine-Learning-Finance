# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:18:52 2017
Fast.ai - Movie Sentiment Dataset
@author: wangx3
"""

import pickle
import numpy as np
model_path = 'data/imdb/models/'
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.preprocessing import image, sequence
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import imdb


idx = imdb.get_word_index()
idx_arr = sorted(idx, key=idx.get)
# Python3 dict.items() replace dict.iteritems() in Python 2.7
idx2word = {v: k for k, v in idx.items()}

file = get_file('imdb_full.pkl', 
                origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl', 
                md5_hash='d091312047c43cf9e4e38fef92437263')
imdb = open(file, 'rb')
(x_train, labels_train), (x_test, labels_test) = pickle.load(imdb)

# truncate vocaburary down to 5000
vocab_size = 5000
seq_len = 500
trn = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in x_train]
test = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in x_test]
# Pad(with zero) or truncate each sentences to make consistent length
trn = sequence.pad_sequences(trn, maxlen=seq_len, value=0)
test = sequence.pad_sequences(test, maxlen=seq_len, value=0)
'''
# Simple model 对于大型的Embedding模型，50dimentions可以捕捉10万以上的词汇量
# 对于5000的词汇量，32维模型足够了. val acc=0.86
model = Sequential([Embedding(vocab_size, 32, input_length=seq_len), 
                    Flatten(), 
                    Dense(100, activation='relu'), 
                    Dropout(0.7), 
                    # 由于labels是[0,1]型的，故用sigmoid
                    Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
model.fit(trn, labels_train, validation_data=(test, labels_test), epochs=2, batch_size=16)
'''

# Conv model
conv1 = Sequential([Embedding(vocab_size, 32, input_length=seq_len, dropout=0.2),
                    Dropout(0.2),
                    Conv1D(64, 5, border_mode='same', activation='relu'),
                    Dropout(0.2),
                    MaxPooling1D(),
                    Flatten(),
                    Dense(100, activation='relu'),
                    Dropout(0.7),
                    Dense(1, activation='sigmoid')])
conv1.summary()
conv1.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
conv1.fit(trn, labels_train, validation_data=(test, labels_test), epochs=2, batch_size=16)


