# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:33:21 2016
5档当日盘口
庄家吸筹
@author: wangx3
"""

import numpy as np
import pandas as pd
import tushare as ts
import datetime as dt

# 读取当日交易股票代码清单
active = ts.get_today_all()
codelist = list(active['code'])
# 创建一个空的DataFrame，基于一个定义好的Columns
t = pd.DataFrame(columns=['name', 'open', 'pre_close', 'price', 'high', 
                          'low', 'bid', 'ask', 'volume', 'amount', 
                          'b1_v', 'b1_p', 'b2_v', 'b2_p', 'b3_v', 'b3_p',
                          'b4_v', 'b4_p', 'b5_v', 'b5_p', 'a1_v', 'a1_p', 
                          'a2_v', 'a2_p', 'a3_v', 'a3_p', 'a4_v', 'a4_p', 
                          'a5_v', 'a5_p', 'date', 'time', 'code'])
a = 0
b = 10
while a < len(codelist):
    df = ts.get_realtime_quotes(codelist[a: b])
    a += 10
    b += 10
    # 添加/合并是不保留原来的index
    # concat()是pd的函数，append是DataFrame的方法
    t = pd.concat([t, df], ignore_index=True)

# 下载数据是Str格式并有空格？
t = t.replace('', '0') # 除去空格
t['b1_v'] = t['b1_v'].astype(np.int32)
t['b2_v'] = t['b2_v'].astype(np.int32)
t['b3_v'] = t['b3_v'].astype(np.int32)
t['b4_v'] = t['b4_v'].astype(np.int32)
t['b5_v'] = t['b5_v'].astype(np.int32)
t['a1_v'] = t['a1_v'].astype(np.int32)
t['a2_v'] = t['a2_v'].astype(np.int32)
t['a3_v'] = t['a3_v'].astype(np.int32)
t['a4_v'] = t['a4_v'].astype(np.int32)
t['a5_v'] = t['a5_v'].astype(np.int32)


# 压单量是托单量的15倍以上
t['bidding'] = t['b1_v'] + t['b2_v'] + t['b3_v'] + t['b4_v'] + t['b5_v']
t['asking'] = t['a1_v'] + t['a2_v'] + t['a3_v'] + t['a4_v'] + t['a5_v']
t = t[(t['asking'] / t['bidding']) > 15]

v = pd.DataFrame(columns=['date', 'code', 'volume', 'amount', 'type'])
date = dt.date.today()
codelist = list(t['code'])
for code in codelist:
    df = ts.get_today_ticks(code)
    df = df[(df['type'] == '买盘')]
    volume = df['volume'].sum()
    amount = df['amount'].sum()
    # 先将要添加的数据构成一个独立的DataFrame，再用append方法添加
    i = pd.DataFrame([[date, code, volume, amount, '买盘']], 
                     columns=['date', 'code', 'volume', 'amount', 'type'])
    v = v.append(i, ignore_index=True)  

# 用Pandas函数读写CSV数据（和该执行文件在同一目录）
# 用Encoding指定编码方案，防止乱码  
# mode=a设定写入方式为添加， index=Fales防止把index也写到CSV文件里
v.to_csv('sample.csv', mode='a', header=False, index=False, encoding='utf-8')
w = pd.read_csv('sample.csv', encoding='utf-8') 
    
    
    
