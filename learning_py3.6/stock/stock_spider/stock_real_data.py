# coding:utf-8
'''
股票的实时爬虫
'''
import tushare as ts
# ts.get_today_all()
df = ts.get_k_data('600848')
print(type(df))
df = ts.top_list()
print(df)
