# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 16:51:18 2016
Scikit-Learn
@author: wangx3
"""

import sklearn.datasets as data
import matplotlib.pyplot as plt
import numpy as np

boston = data.load_boston()
plt.scatter(boston.data[:, 5], boston.target, color='r')

plt.figure()
iris = data.load_iris()
features = iris['data']
feature_names = iris['feature_names']
target = iris['target']
for t, marker, c in zip(range(3), '>ox', 'rgb'):
    plt.scatter(features[target == t, 0],
                features[target == t, 1],
                marker = marker,
                c = c)
