# coding:utf-8

'''
使用线性回归彩票销售进行销量的预测
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import linear_model

# 获取销售数据
lottery_sale_all = pd.read_csv('/home/dushao/桌面/all_sale_DF.csv')
lottery_sale_bj = lottery_sale_all[lottery_sale_all['0'] == '北京'].iloc[:, 2: 9]
print(lottery_sale_bj)
lottery_sale_bj.plot()
pyplot.show()
# print(lottery_sale_bj.corr()) 协防差矩阵

# 将销售的数据的下一期作为预测值，将上一期的其他的数据作为变量进进行合并，之后进行预测
lottery_sale_bj_next_draw = lottery_sale_bj.iloc[:, [0]].drop([0]).reset_index(drop=True)

lottery_sale_forecast_feature = lottery_sale_bj.iloc[:, 0:9].drop([14301]).reset_index(drop=True)
train_data = pd.concat([lottery_sale_bj_next_draw, lottery_sale_forecast_feature], axis=1,)
print(train_data.corr())





























