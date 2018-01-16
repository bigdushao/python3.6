# coding：utf-8
'''
pandas和numpy数组之间的相互转化
'''

import numpy as np
import pandas as pd

# 输入的数据data为list
data = [[2000, 'Ohino', 1.5],
        [2001, 'Ohino', 1.7],
        [2002, 'Ohino', 3.6],
        [2003, 'nevada', 3.7],
        [2004, 'nevada', 4.7]] # list
print(type(data))

# 序列 list to series
ser = pd.Series(data, index=['one', 'tow', 'three', 'four', 'five'])
print(ser[0:1])

# list to data frame
df = pd.DataFrame(data,
                  index=['one', 'tow', 'three', 'four', 'five'],
                  columns=['year', 'state', 'pop'])
# foo = pd.Series.as_matrix(ser) series to ndarray
foo = ser.as_matrix() # series to ndarray
print(foo)

foo = pd.DataFrame.as_matrix(df) # data frame to ndarray
foo = df.as_matrix() # data frame to ndarray
foo = df.values # data frame to ndarray
foo = np.array(df) # data frame to ndarry

print(foo)
foo = pd.DataFrame.as_matrix(df)[:, 2] # data frame to ndarray
foo = df.as_matrix(['pop']) # data frame to ndarray
print(foo)

'''
将numpy中的数组转化为data frame
'''

data = [[2000, 'Ohino', 1.5],
        [2001, 'Ohino', 1.7],
        [2002, 'Ohino', 3.6],
        [2003, 'nevada', 3.7],
        [2004, 'nevada', 4.7]]
data = np.array(data) # ndarray

df = pd.DataFrame(data,
                  index=['one', 'two', 'three', 'four', 'five'],
                  columns=['year', 'state', 'pop'])
print(df)

'''
dict to data frame
'''
# dict
data = {'state': ['ohino', 'ohino', 'ohino', 'nevada', 'nevada'],
        'year': [2000, 2001, 2002, 2001, 2001],
        'pop': [1.5, 1.6, 1.7, 1.8, 1.9]} # dict

ser = pd.Series(data, index=['one', 'two', 'three', 'four', 'five'])

df = pd.DataFrame(data)