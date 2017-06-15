# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 21:01:10 2017
Quants
@author: 榴莲男爵
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts

mpl.style.use('seaborn-whitegrid')

def sta001(iRate, nYear, aDeposit):
    total = np.fv(iRate, nYear, -aDeposit, -aDeposit)
    return round(total)

d05 = [sta001(0.05, x, 1.4) for x in range(40)]
d10 = [sta001(0.10, x, 1.4) for x in range(40)]
d15 = [sta001(0.15, x, 1.4) for x in range(40)]
d20 = [sta001(0.20, x, 1.4) for x in range(40)]

df = pd.DataFrame(columns=['d05','d10','d15','d20'])
df['d05'] = d05
df['d10'] = d10
df['d15'] = d15
df['d20'] = d20

#df.plot()

dfBasic = ts.get_stock_basics()
dfHist = ts.get_hist_data('600848')
dfSina = ts.get_sina_dd('600848', date='2017-04-20')
dfHist['close'].plot()




