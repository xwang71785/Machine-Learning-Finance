# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:09:37 2017
Build RNN
@author: wangx3
"""

import numpy as np
from keras.models import Sequential, Model
from keras.utils.data_utils import get_file
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input, Embedding, Reshape, LSTM, Bidirectional
from keras.layers.merge import add
from keras.optimizers import SGD, RMSprop, Adam


file = 'nietzsche.txt'
path = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'
sample = get_file(file, origin=path)
text = open(sample).read()
# print('corpus length:', len(text))
chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
# print('', vocab_size)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
idx = [char_indices[c] for c in text]

def embedding_input(name, n_in, n_out):
    inp = Input(shape=(1,), dtype='int64', name=name)
    emb = Embedding(n_in, n_out, input_length=1)(inp)
    return inp, Flatten()(emb)

'''
# 3 char model
cs = 3
c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])
y = np.stack(c4_dat[:-2])
# the size of embedding matrix
n_fac = 42
c1_in, c1 = embedding_input('c1', vocab_size, n_fac)
c2_in, c2 = embedding_input('c2', vocab_size, n_fac)
c3_in, c3 = embedding_input('c3', vocab_size, n_fac)
# build and train model
n_hidden = 256
dense_in = Dense(n_hidden, activation='relu')
c1_hidden = dense_in(c1)    # green arrow
dense_hidden = Dense(n_hidden, activation='tanh')
c2_dense = dense_in(c2)    # green arrow
hidden_2 = dense_hidden(c1_hidden)     # orange arrow
c2_hidden = add([c2_dense, hidden_2])    # merge() is relaced by merge.add()
c3_dense = dense_in(c3)
hidden_3 = dense_hidden(c2_hidden)
c3_hidden = add([c3_dense, hidden_3])
dense_out = Dense(vocab_size, activation='softmax')
c4_out = dense_out(c3_hidden)
model = Model([c1_in, c2_in, c3_in], c4_out)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
model.optimizer.lr=0.01
model.fit([x1, x2, x3], y, batch_size=64, epochs=1)
model.optimizer.lr=0.0000001
model.fit([x1, x2, x3], y, batch_size=64, epochs=1)
'''

cs = 8
c_in_dat = [[idx[i+n] for i in range(0, len(idx)-1-cs, cs)] for n in range(cs)]
c_out_dat = [idx[i+cs] for i in range(0, len(idx)-1-cs, cs)]
xs = [np.stack(c[:-2]) for c in c_in_dat]
print(len(xs), xs[0].shape)
y = np.stack(c_out_dat[:-2])
n_fac = 42
c_ins = [embedding_input('c'+str(n), vocab_size, n_fac) for n in range(cs)]
n_hidden = 256
dense_in = Dense(n_hidden, activation='relu')
dense_hidden = Dense(n_hidden, activation='relu', kernel_initializer='identity')
dense_out = Dense(vocab_size, activation='softmax')
hidden = dense_in(c_ins[0][1])
for i in range(1, cs):
    c_dense = dense_in(c_ins[i][1])
    hidden = dense_hidden(hidden)
    hidden = add([c_dense, hidden])
c_out = dense_out(hidden)
model = Model([c[0] for c in c_ins], c_out)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
model.optimizer.lr=0.000001
model.fit(xs, y, batch_size=64, epochs=12)



