# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:50:56 2016

@author: wangx3
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt

start = dt.date(2016, 1, 1)
end = dt.datetime.today()
code = 'AAPL'
week = 5
biweek = 10
month = 21
year = 252

df = web.get_data_yahoo(code, start, end)
vwap = np.average(df['Adj Close'], weights=df['Volume'])   #加权平均值
highest = np.max(df['High'])
lowest = np.min(df['Low'])
median = np.median(df['Open'])    # 中位数（对于个数是偶数的情况，去中间两个数的平均数）
variance = np.var(df['Close'])
# 收益
df['Return'] = df['Adj Close'].diff()
# 对数收益率
df['Log_Close'] =np.log(df['Adj Close'])
df['Log_Ret'] = df['Log_Close'].diff()
# 移动均线
df['Week'] = df['Adj Close'].rolling(window=week).mean()
df['Month'] = df['Adj Close'].rolling(window=month).mean()

# 波动率
df['Volatility'] = df['Log_Ret'].rolling(window=month).std() * np.sqrt(month)
# 布林带
df['Variance'] = df['Adj Close'].rolling(window=week).var()
df['UpperBB'] = df['Week'] + 2 * df['Variance']
df['LowerBB'] = df['Week'] - 2 * df['Variance']

# 相关性分析

# OBV
df['Return'].dropna(inplace=True)  
df['Sign'] = np.sign(df['Return'])
df['OBV'] = df['Volume'] * df['Sign']









