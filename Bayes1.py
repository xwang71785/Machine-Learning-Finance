# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:08:55 2017

@author: wangx3
"""

import pymc3 as pm

with pm.Model() as model:
    dice1 = pm.DiscreteUniform('dice1', lower=1, upper=6)
    dice2 = pm.DiscreteUniform('dice2', lower=1, upper=6)
    result = pm.Binomial('result', dice1+dice2)
    step = pm.Metropolis()
    trace = pm.sample(25000, step=step)
    burned_trace = trace[2500:]

    