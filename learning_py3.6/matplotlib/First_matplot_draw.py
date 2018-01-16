# coding:utf-8
import pandas as pd
from pylab import *

# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# C, S = np.cos(X), np.sin(X)
# # plot(X, C)
# # plot(X, S)
# # 默认的配置
# plot(X, C, color='blue', linewidth=1.0, linestyle='-', label='cosine')
# plot(X, S, color='green', linewidth=1.0, linestyle='-', label='sine')
#
# # 添加图列
# legend(loc='upper left')
# # 设置横轴上下限
# xlim(-4.0, 4.0)
# # 设置横轴记号
# xticks(np.linspace(-4, 4, 9, endpoint=True))
# # 设置纵轴的上下限
# ylim(-1.0, 1.0)
# # 设置纵轴记号
# yticks(np.linspace(-1, 1, 5, endpoint=True))

# 以分辨率72来保存图片
# savefig("exercice_2.png", dpi=72)
# 在屏幕上显示
# show()

lottery_sale = pd.read_csv('/home/dushao/桌面/all_sale_DF.csv')
bj = lottery_sale[(lottery_sale['0'] == '北京')]
bj_sale = bj.loc[:, ['10', '1']]
bj_sale['1'] = bj_sale['1'].astype('int')
bj_sale['10'] = bj_sale['10'].astype('str')

hn = lottery_sale[(lottery_sale['0'] == '河南')]
hn_sale = hn.loc[:, ['10', '1']]
hn_sale['1'] = hn_sale['1'].astype('int')
hn_sale['10'] = hn_sale['10'].astype('str')
hn_sale.plot(x='10', y='1')
bj_sale.plot(x='10', y='1')
print(hn.head())
print(bj.head())
show()