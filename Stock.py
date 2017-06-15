#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

Topic: Python for Finance.
Author: Wang Xin
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import matplotlib.figure as fig
import datetime as dt

week = 5
biweek = 10
month = 42
year = 252
# Pandas对日期格式的要求dt.date(),也接受字符串格式‘6/30/2016’
# Tushare接受字符串格式的日期 ‘2016-11-02’
# Matplotlib接受tuple格式的日期(2016, 11, 2)
start = dt.date(2014, 1, 1)
end = dt.date.today()
ticker = 'AMAT'
stocklist = ('AMAT', 'ETFC', 'VLO', 'CRZO', 'CST', 'XLNX', 'BABA')

# DataReader返回DataFrame，以日期为index,顺序ohlcv

for ticker in stocklist:
    df = web.get_data_yahoo(ticker, start, end)
    df['Week'] = df['Adj Close'].rolling(window=week).mean()
    df['Biweek'] = df['Adj Close'].rolling(window=biweek).mean()
    df['Month'] = df['Adj Close'].rolling(window=month).mean()
    df['Year'] = df['Adj Close'].rolling(window=year).mean()
    fig, axes = plt.subplots(2, sharex=True, figsize=(16, 9))
    df[['Adj Close', 'Month', 'Year']].plot(ax=axes[0], grid=True, title=ticker)
    df['Volume'].plot(ax=axes[1], grid=True, title='Volume')
# plt.show()

# candlestick不接受 DataFrame格式的数据
# matplotlib.quotes函数返回list.日期格式是float的
# list indices must be integers or slices, not tuple要用np.array转换
quotes = np.array(mpf.quotes_historical_yahoo_ohlc(ticker, start, end))
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 9))
mpf.candlestick_ohlc(ax1, quotes, width=0.8, colorup='r', colordown='g')
ax1.plot(df['Month'])
ax1.set_title(ticker)
ax1.xaxis_date()
ax1.grid(True)
plt.bar(quotes[:, 0] - 0.25, quotes[:, 5], width=0.6)
ax2.set_ylabel('volume')
ax2.grid(True)
# 调整x轴标签的角度
plt.setp(plt.gca().get_xticklabels(), rotation=60)
plt.show()
