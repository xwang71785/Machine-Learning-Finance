#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:14:23 2016
Topic: TuShare
Author: wangx3
"""


import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import matplotlib.dates as mdt
import datetime as dt

# TuShare用str作为日期的格式
stocklist = ('002621', '300070', '600848', '001696')
ticker = '001696'
start = '2016-01-01'
end = '2016-12-30'
'''
df1 = ts.get_hist_data(ticker, start, end)
df2 = ts.get_k_data(ticker, start, end)

# df3 = ts.get_tick_data('300070', '2016-11-25')

df2['date'] = pd.to_datetime(df2['date'])
df2['date'] = mdt.date2num(df2['date'].astype(dt.date))

quotes = np.array(df2)
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 9))
mpf.candlestick_ochl(ax1, quotes, width=0.6, colorup='r', colordown='g')
ax1.grid(True)
ax1.xaxis_date()
plt.bar(quotes[:, 0], quotes[:, 5], width=0.6)
ax2.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=90)
plt.show()
'''
df = ts.global_realtime()
