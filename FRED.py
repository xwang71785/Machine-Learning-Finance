# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:51:07 2016

Pandas_DataReader FRED
@author: wangx3
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt

start = dt.datetime(2014, 1, 1)
end = dt.datetime.today()
index = pd.DataFrame(columns=['Treasury', 'Libor'])

# 10-Year Treasury Constant Maturity Rate
dgs10 = web.get_data_fred('DGS10', start, end)
# Overnight London InterBank Offered Rate, based on USD (USDONTD156N)
# 1-Month London InterBank Offered Rate, based on USD (USD1MTD156N)
libor = web.get_data_fred('USDONTD156N', start, end)
# Gold Fixing Price AM in London Bullion Market, Based on USD
gold = web.get_data_fred('GOLDAMGBD228NLBM', start, end)
# China/US Foreign Exchange Rate
cnus = web.get_data_fred('DEXCHUS', start, end)
# Crude Oil Price, West Taxes Intermediate
oil = web.get_data_fred('DCOILWTICO', start, end)

index['Treasury'] = dgs10['DGS10']
index['Libor'] = libor['USDONTD156N']
index['CNYUSD'] = cnus['DEXCHUS']
index['Gold'] = gold['GOLDAMGBD228NLBM']
index['Oil'] = oil['DCOILWTICO']
index.plot(subplots=True, figsize=(16, 10), grid=True, x_compat=True)
plt.setp(plt.gca().get_xticklabels(), rotation=70)
plt.show()
