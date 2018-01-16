# coding:utf-8

'''
首先模型进行评估，解释变量为图片的尺寸，网址链接的里的单词，以及图片标签周围的单词，响应变量就是图片的类型。
前三个特征表示宽度，高度，图像纵横对比，剩下的特征是文本变量的二元频率
使用网格搜索来确定决策树模型最大最优评价效果
'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# 读取数据，源数据是zip
df = pd.read_csv("", header=True)
explanatory_variable_columns = set(df.columns.values)
response_variable_columns = df[len(df.columns.values) - 1]
# the last column describes the targets
explanatory_variable_columns.remove(len(df.columns.values) - 1)
y = [1 if e == 'ad.' else 0 for e in response_variable_columns]
x = df.loc[:, list(explanatory_variable_columns)]

x.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(x, y)

pipeline = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])

parameters = {
    'clf_max_depth': (150, 155, 160),
    'clf_max_samples_split': (1, 2, 3),
    'clf_min_sample_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(X_train, Y_train)

best_paraments = grid_search.best_params_ = grid_search.best_estimator_.get_params()
