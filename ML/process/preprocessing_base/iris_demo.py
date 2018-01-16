# coding:utf-8

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 导入iris数据集
iris = load_iris()
# 特征矩阵（有四个特征）X
iris.data
#目标向量 b 数值为0,1,2代表三种不同的花
iris.target







