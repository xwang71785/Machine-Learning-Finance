# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:03:13 2017
Fast.ai
@author: wangx3
"""
import math, sys, os
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import ipywidgets.widgets as wdt

np.set_printoptions(precision=4, linewidth=100)

def lin(a, b, x): return a * x + b

a = 3.
b = 8.
n = 30
x = np.random.random(n)
y = lin(a, b, x)

def sse(y, y_pred): return ((y - y_pred) ** 2).sum()
def loss(y, a, b, x): return sse(y, lin(a, b, x))
def avg_loss(y, a, b, x): return np.sqrt(loss(y, a, b, x) / n)

a_guess = -1
b_guess = 1
print(avg_loss(y, a_guess, b_guess, x))

lr = 0.01
def upd():
    global a_guess, b_guess
    y_pred = lin(a_guess, b_guess, x)
    dydb = 2 * (y_pred - y)
    dyda = x * dydb
    a_guess -= lr * dyda.mean()
    b_guess -= lr * dydb.mean()
    
fig = plt.figure(dpi=100, figsize=(9, 6))
plt.scatter(x, y)
line, = plt.plot(x, lin(a_guess, b_guess, x))  
#plt.close() 

def animate(i):
    line.set_ydata(lin(a_guess, b_guess, x))
    for i in range(10): upd()
    return line

ani = anm.FuncAnimation(fig, animate, np.arange(0, 40), interval=500)
ani






    
    




