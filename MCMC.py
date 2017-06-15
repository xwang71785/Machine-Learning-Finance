# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:39:00 2017
MCMC
@author: wangx3
"""

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

plt.plot(figsize=(16, 9))
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

jet = plt.cm.jet
fig = plt.figure()
x = y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)

plt.subplot(121)
uni_x = st.uniform.pdf(x, loc=0, scale=5)
uni_y = st.uniform.pdf(y, loc=0, scale=5)
M = np.dot(uni_x[:, None], uni_y[None, :])
im = plt.imshow(M, interpolation='none', origin='lower', cmap=jet, vmax=1,
                vmin=-.15, extent=(0, 5, 0, 5))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()
