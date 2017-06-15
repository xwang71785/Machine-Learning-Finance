# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 22:25:17 2016
跳档买入卖出
@author: admin
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.dates as mdt
import matplotlib.pylab as plt
import matplotlib.finance as mpf
import tushare as ts

ccode = '001696'
startday = '2016-06-01'
endday = '2016-12-31'
histd = ts.get_k_data(code=ccode, start=startday, end=endday, ktype='D')

peroid = histd['date']
# convert str to date
histd['date'] = pd.to_datetime(histd['date'])
# convert date to float days
histd['date'] = mdt.date2num(histd['date'].astype(dt.date))

rate = np.zeros((len(peroid), 2))
x = 0
for vdate in peroid:    # 此处的histd['date']必须是格式转换之前的
    tick = ts.get_tick_data(code=ccode, date=vdate)
    if len(tick) > 4:     # 判断返回的数据是否为空
        tick['change'].replace('--', np.nan, inplace=True)  # 把空值--替换成np.nan
        tick.dropna(inplace=True)   # 用dropna把含有nan的行删除
        tick['price'] = tick['price'].astype(float)
        tick['change'] = tick['change'].astype(float)
        tick['p_change'] = tick['change'] / tick['price']
        tick_up = tick[(tick['p_change'] > 0.003) & (tick['type'] == '买盘')]
        tick_down = tick[(tick['p_change'] < -0.003) & (tick['type'] == '卖盘')]
        rate[x, 0] = 100 * tick_up['amount'].sum() / (tick['amount'].sum() + 0.001)
        rate[x, 1] = 100 * tick_down['amount'].sum() / (tick['amount'].sum() + 0.001)
    else:
        print('Error on date', vdate)
        rate[x, 0] = 0
        rate[x, 1] = 0
    x = x + 1
histd['jump'] = rate[:, 0]
histd['drop'] = rate[:, 1]
hista = np.array(histd)

# 转换DataFrame为Numpy.Array(网上有重新构建tuple的方法,我觉得转换成Numpy Array更直接)
# 利用hista画出蜡烛图
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 9))
fig.subplots_adjust(bottom=0.2)
mpf.candlestick_ochl(ax1, hista, width=0.8, colorup='r', colordown='g')
ax1.xaxis_date()
ax1.grid(True)
plt.bar(hista[:, 0]-0.3, hista[:, 7], width=0.4, color='r')
plt.bar(hista[:, 0], hista[:, 8], width=0.4, color='g')
ax2.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=70)
plt.show()
