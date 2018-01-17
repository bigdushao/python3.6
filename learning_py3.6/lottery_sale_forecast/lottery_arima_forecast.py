# coding:utf-8

'''
使用python arima 进彩票销售进行销量的预测
'''

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from datetime import datetime
import matplotlib.pylab as plt
# 获取销售数据
lottery_sale_all = pd.read_csv('/home/dushao/桌面/all_sale_DF.csv')
lottery_sale_bj = lottery_sale_all[lottery_sale_all['0'] == '北京'].iloc[:, 2]
lottery_sale_draw = lottery_sale_all[lottery_sale_all['0'] == '北京'].iloc[:, 11]

def datelist(beginDate, endDate):
    # beginDate, endDate是形如‘20160601’的字符串或datetime格式
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

date = pd.Series(datelist('20170101', '20170606')).head(140)

# print(lottery_sale_bj.tail(140))
# 2017年的140期销售数据
series = lottery_sale_bj.tail(140)
series.index = date
# print(series)
# series.plot()
# 观察发现数据是不平稳的
# lottery_sale_bj.tail(140).plot()
#
# pyplot.show()
# 使用三期的均值进行平滑处理
# rolmean = pd.rolling_mean(series, window=3)
# rolmean.plot()


# 进行一阶差分处理 对进行均值化处理后的数据进行一阶差分
# diff_1 = rolmean - rolmean.shift().plot()
# # 对原始数据进行差分处理
# diff_1 = series - series.shift()
# diff_1.dropna(inplace=True)

# 取对数后的差分处理
log_series = np.log(series)
diff_1 = log_series - log_series.shift()
diff_1.dropna(inplace=True)
# diff_1.plot()
# pyplot.show()



def t_analysis():
    # Plot ACF:
    lag_acf = acf(diff_1, nlags=20)
    lag_pacf = pacf(diff_1, nlags=20, method='ols')
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(diff_1)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(diff_1)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(diff_1)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(diff_1)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    plt.show()

model_1 = ARIMA(diff_1, order=(2, 1, 0))
results_ARIMA = model_1.fit(disp=-1)
plt.plot(diff_1)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues-diff_1)**2))

# step-3:将模型代入原数据进行预测
# 因为上面的模型的拟合值是对原数据进行稳定化之后的输入数据的拟合，所以需要对拟合值进行相应处理的逆操作，使得它回到与原数据一致的尺度
# ARIMA拟合的其实是一阶差分ts_log_diff，predictions_ARIMA_diff[i]是第i个月与i-1个月的ts_log的差值。
# 由于差分化有一阶滞后，所以第一个月的数据是空的，
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
# 累加现有的diff，得到每个值与第一个月的差分（同log底的情况下）。
# 即predictions_ARIMA_diff_cumsum[i] 是第i个月与第1个月的ts_log的差值。
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# 先ts_log_diff => ts_log=>ts_log => ts
# 先以ts_log的第一个值作为基数，复制给所有值，然后每个时刻的值累加与第一个月对应的差值(这样就解决了，第一个月diff数据为空的问题了)
# 然后得到了predictions_ARIMA_log => predictions_ARIMA
predictions_ARIMA_log = pd.Series(log_series.ix[0], index=log_series.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.figure()
plt.plot(series)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA - series)**2)/len(series)))
plt.show()


def analysis_1(series):
    rolmean = series.rolling_mean(series, window=3)
    rolmean.plot()
    diff_rolmean = rolmean - rolmean.shift()
    diff_rolmean.plot()

def analysis_2(series):
    org_diff = series - series.shift()
    org_diff.plot()

def analysis_3(series):
    log_seris = np.log(series)
    # log_seris.plot()
    log_seris_diff = log_seris - log_seris.shift()
    log_seris_diff.plot()


