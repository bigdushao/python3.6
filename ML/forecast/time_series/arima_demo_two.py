# coding:utf-8
from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot  as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

data = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
        6337, 11649, 11652, 10310, 12043, 7937, 6476, 9662, 9570, 9981, 9331, 9449, 6773, 6304, 9355,
        10477, 10148, 10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095, 7707, 10767,
        12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683, 12603, 11495, 13670, 11337, 10232,
        13261, 13230, 15535, 16837, 19598, 14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248,
        9543, 12872, 13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085, 14722,
        11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]

data = pd.Series(data)
data.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))
data.plot(figsize=(12, 8))
# 一阶差分
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111)
diff1 = data.diff(1)
diff1.plot(ax=ax1)
# 二阶差分
fig = plt.figure(figsize=(12,8))
ax2 = fig.add_subplot(111)
diff2 = data.diff(2)
diff2.plot(ax=ax2)

dta = data.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

plt.show()