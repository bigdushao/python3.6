# coding:utf-8

'''
使用python做时间序分析，使用的是ARMA模型进行的分析
'''

import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt

# 读取数据，pd.read_csv 默认生成DataFrame对象，将其转化成Series对象
df = pd.read_csv("/home/dushao/work/python3.6/ML/forecast/time_series/AirPassengers.csv", encoding='utf-8', index_col='Month')
df.index = pd.to_datetime(df.index) # 将字符串索引转化成时间索引
print(df.head())
ts = df['AirPassengers'] # 生成pd.series对象

# 查看数据格式
print(ts.head())
print(ts.head().index)
# 查看某日的值既可以使用字符串作为索引，又可以直接使用时间对向作为索引
print(ts['1949-01-01'])
print(ts[datetime(1949, 1, 1)])
print(ts['1949']) # 查看一年的数据。类似正则匹配
print(ts['1949-1': '1949-6']) # 时间索引的切片操作起点和尾部都是包含的，这点与数值索引有所不同

'''
在大数定理和中心定理中要求样本同分布（这里同分布等价于时间序列中的平稳性），
而我们的建模过程中有很多都是建立在大数定理和中心极限定理的前提条件下的，
如果它不满足，得到的许多结论都是不可靠的。以虚假回归为例，当响应变量和输入变量都平稳时，我们用t统计量检验标准化系数的显著性。
而当响应变量和输入变量不平稳时，其标准化系数不在满足t分布，这时再用t检验来进行显著性分析，导致拒绝原假设的概率增加，即容易犯第一类错误，
从而得出错误的结论

严平稳的条件只在理论上存在，现实中使用比较多的是宽平稳
宽平稳也成为弱平稳，或者二阶平稳(均值和方差平稳)
    常数均值
    常数方差
    常数自协防差
    
平稳性处理
    序列的平稳性是时间序列分析的前提条件，我们需要对不平稳的序列进行处理将其转化为平稳的序列
    对数变化：对数变换主要是为了减小数据的振动幅度，使其线性规律更加明显（我是这么理解的时间序列模型大部分都是线性的，
        为了尽量降低非线性的因素，需要对其进行预处理，也许我理解的不对）。
        对数变换相当于增加了一个惩罚机制，数据越大其惩罚越大，数据越小惩罚越小。
        这里强调一下，变换的序列需要满足大于0，小于0的数据不存在对数变换。
    平滑法：根据平滑技术的不同，平滑法具体分为移动平均法和指数平均法。
           移动平均即利用一定时间间隔内的平均值作为某一期的估计值，而指数平均则是用变权的方法来计算均值
    差分:剔除周期性因素的方法。对周期间隔的数据进行线性求减
    分解:
'''
# python 判断时序数据稳定性

def arima_stationarity(timeseries):
    '''
    Rolling statistic-- 即每个时间段内的平均的数据均值和标准差情况。

    Dickey-Fuller Test -- 这个比较复杂，大致意思就是在一定置信水平下，对于时序数据假设 Null hypothesis: 非稳定。
    if 通过检验值(statistic)< 临界值(critical value)，则拒绝null hypothesis，即数据是稳定的；反之则是非稳定的。
    :param timeseries: 
    :return: 
    '''
    # 以一年为一个窗口，每一个时间t的值由它前面12个月(包括自己)的均值代替，标准差同理。
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # plot rolling statistics
    fig = plt.figure()
    fig.add_subplot()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='rolling mean')
    std = plt.plot(rolstd, color='black', label='Rolling standard deviation')

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey-Fuller test:
    print('Reuslts of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    # dftest的输出前一项一次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value
    print(dfoutput)
# arima_stationarity(ts)
# 数据的rolling均值/标准差具有越来越大的趋势，是不稳定的。且DF-test可以明确的指出，在任何置信度下，数据都不是稳定的
# plt.show()

'''
让数据变得不稳定的原因主要有俩:
    1.趋势(trend)-数据随着时间变化。比如升高或者降低
    2.季节性(seasonality)-数据在特定的时间段内变动，比如说节假日，或者活动导致数据的异常。
    由于原数据值域范围比较大，为了缩小值域，同时保留其他信息，常用的方法是对数化，取log。
    检测和取出趋势通常有三种方法:
    聚合 : 将时间轴缩短，以一段时间内星期/月/年的均值作为数据值。使不同时间段内的值差距缩小。
    平滑： 以一个滑动窗口内的均值代替原来的值，为了使值之间的差距缩小
    多项式过滤：用一个回归模型来拟合现有数据，使得数据更平滑。
'''
# 使用moving_average比原值平滑很多
# 做差
ts_log = np.log(ts)
moving_avg = pd.rolling_mean(ts_log, 12)
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
# arima_stationarity(ts_log_moving_avg_diff)
# 可以看到，做了处理之后的数据基本上没有了随时变化的趋势，DFTest的结果显示95%的置信度下，数据是稳定的。
# 该方法是将所有的时间平等看待，而在许多情况下，可以认为越近的时刻越重要，所以引入指数加权移动平均--Exponentially-weighted moving average
#   pandas中通过ewma()函数提供了此功能
# plt.show()

# halflife的值决定了衰减因子alpha: alpha = 1 - exp(log(0.5) / halflife)
expweighted_avg = pd.ewma(ts_log, halflife=12)
ts_log_ewma_diff = ts_log - expweighted_avg
# arima_stationarity(ts_log_ewma_diff)
# 可以看到相比普通的Moving Average，新的数据平均标准差更小了。而且DFtest可以得到结论：数据在99%的置信度上是稳定的。
# plt.show()

'''
检测和去除季节性
    差分化:以特定滞后数目的时刻的值的作差
    分解:对趋势和季节性分别建模在移除它们
'''
# differencing -- 差分
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
# arima_stationarity(ts_log_diff)
# 可以看出相比MA方法，Differencing方法处理后的数据的均值和方差的在时间轴上的振幅明显缩小了。DFtest的结论是在90%的置信度下，数据是稳定的。
# plt.show()

# Decomposing-分解
def decompose(timeseries):
    # 返回包含三个部分trend(趋势部分), seasonal(季节性部分), residual(残留部分)
    '''
    将original数据 拆分成了三份。
    Trend数据具有明显的趋势性，
    Seasonality数据具有明显的周期性，
    Residuals是剩余的部分，可以认为是去除了趋势和季节性数据之后，稳定的数据，
    :param timeseries: 
    :return: 
    '''
    decomposition = seasonal_decompose(timeseries)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

    return trend, seasonal, residual

trend, seasonal, residual = decompose(ts_log)
residual.dropna(inplace=True)
# arima_stationarity(residual)
# 数据的均值和方差趋于常数，几乎无波动(看上去比之前的陡峭，但是要注意他的值域只有[-0.05,0.05]之间)，所以直观上可以认为是稳定的数据。
# 另外DFtest的结果显示，Statistic值原小于1%时的Critical value，所以在99%的置信度下，数据是稳定的。
# plt.show()

'''
对序列数据进行预测
step1： 通过ACF,PACF进行ARIMA（p，d，q）的p，q参数估计
    由前文Differencing部分已知，一阶差分后数据已经稳定，所以d=1。
    所以用一阶差分化的ts_log_diff = ts_log - ts_log.shift() 作为输入。
    等价于
    yt=Yt−Yt−1
    作为输入。
'''
# acf,pacf的图像
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
# Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

# Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
# 图中，上下两条灰线之间是置信区间，p的值就是ACF第一次穿过上置信区间时的横轴值。
# q的值就是PACF第一次穿过上置信区间的横轴值。所以从图中可以得到p=2，q=2。
# plt.show()

# step-2:得到参数估计值p, d, q之后，生成模型ARIMA(p, d, q) 蓝线是输入值，红线是模型的拟合值，RSS的累计平方误差。
# model1 ARIMA(2, 1, 0)
model_1 = ARIMA(ts_log, order=(2, 1, 0))
results_AR = model_1.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_AR.fittedvalues-ts_log_diff)**2))
# model2 ARIMA(0, 1, 2)
model_2 = ARIMA(ts_log, order=(0, 1, 2))
results_MA = model_2.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

# model3 ARIMA(2, 1, 2)
model_3 = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model_3.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
# plt.show()

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
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.figure()
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()