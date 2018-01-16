# coding:utf-8

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

'''
使用随机森林回归算法进行预测计算
'''

X, Y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, Y)
# 默认是用的参数
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
#                       max_features='auto', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#                       oob_score=False, random_state=0, verbose=0, warm_start=False)


print(regr.feature_importances_)

print(regr.predict([[0, 0, 0, 0], [1, 1, 1, 1]]))
print(type(regr.decision_path(X)[1]))
