# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:10:07 2017
Quickstart to PyMC3
@author: wangx3
"""

import pymc3 as pm
import theano.tensor as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)

num_feature = 10
num_observed = 1000

alpha_a = np.random.normal(size=1)
betas_a = np.random.normal(size=num_feature)
X_train = np.random.normal(size=(num_observed, num_feature))
y_a = alpha_a + np.sum(betas_a[None,:] * X_train, 1) + np.random.normal(size=(num_observed))

lin_reg_model = pm.Model()
with lin_reg_model:
    alpha = pm.Normal('alpha', mu=0, tau=10.**2, shape=(1))
    betas = pm.Normal('belta', mu=0, tau=10.**-2, shape=(1, num_feature))
    s = pm.HalfNormal('s', tau=1)
    temp = alpha + T.dot(betas, X_train.T)
    y = pm.Normal('y', mu=temp, tau=s**-2, observed=y_a)
    
with lin_reg_model:
    step = pm.NUTS()
    nuts_trace = pm.sample(2000, step)
    
pm.traceplot(nuts_trace[100:])
    
    
    
    
    
    
    
    
    