# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 21:25:55 2016
Daily
换手率大于3的股票是否有跳档买入或卖出超过20%
@author: wangx3
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.dates as mdt
import matplotlib.pylab as plt
import matplotlib.finance as mpf
import tushare as ts

date = dt.date.today()
df = ts.get_today_all()
df_filtered = df[df['turnoverratio']>3]
clist = df_filtered['code']

rate = np.zeros((len(df_filtered), 2))
x = 0
for code in clist:
    tick = ts.get_today_ticks(code)
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
        print('Error in stock', code, 'on', date)
        rate[x, 0] = 0
        rate[x, 1] = 0
    x += 1
df['jump'] = rate[:, 0]
df['drop'] = rate[:, 1]
