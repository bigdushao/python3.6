# coding:uft-8

"""
决策树中的CART决策树
"""
from collections import defaultdict
import numpy as np

class TreeNode(object):
    """
    决策树节点
    """
    def __init__(self, **kwargs):
        """
        attr_index : 属性编号
        attr: 属性值
        label: 类别
        left_child: 左子节点
        right_child: 右子节点
        :param kwargs: 
        """
        self.attr_index = kwargs.get('attr_index')
        self.attr = kwargs.get('attr')
        self.label = kwargs.get('label')
        self.left_child = kwargs.get('left_child')
        self.right_child = kwargs.get('right_child')


class DecisionTreeClassifier(object):
    """
    决策树分类器
    使用的是分类与回归树(classification and regression and tree)
    """
    def __init__(self):
        '''
        决策树根节点
        '''
        self.root = None

    def gini(self, cluster):
        '''
        求给定数据集的一个子集
        :param cluster: 训练集的一个子集
        :return: 数据集的基尼系数
        '''
        p = defaultdict(int)
        for line in cluster:
            p[line[-1]] += 1
        temp = 1.0
        for k, v in p.items():
            temp -= (v / len(cluster)) ** 2
        return temp

    def gini_index(self, cluster, attr_index):
        '''
        返回给定列标号下的最有切分属性和该属性的基尼指数
        :param cluster: 训练集的一个子集
        :param attr_index: 特征编号(第N个特征)
        :return: 第N个特征的特征值，该值的基尼指数
        '''
        p = defaultdict(list)
        for line in cluster:
            p[line[attr_index]].append(line)
        attr_gini = {}
        for k, v in p.items():
            els = []
            for k1, v1 in p.items():
                if k1 == k:
                    continue
                els.extend(v)
            count = (self.gini(v) * len(v) + self.gini(els) * len(els)) / len(cluster)
            attr_gini[k] = count
        attr = min(attr_gini, key=attr_gini.get)
        return attr, attr_gini[attr]

    def devide_set(self, cluster, index, attr):
        '''
        将给定集合切分为两部分返回，第index个特征的特征值等于attr的为一组，不等于attr的为一组
        :param cluster: 给定集合(为训练的一个子集)
        :param index: 特征编号
        :param attr: 特征值
        :return: 左半部分， 右半部分
        '''
        left = []
        right = []
        for line in cluster:
            if line[index] == attr:
                left.append(line)
            else:
                right.append(line)
        return np.array(left), np.array(right)

    def get_best_index(self, cluster, attr_indexs):
        '''
        求给定切分点集合中的最佳切分点和其对应的最佳切分变量
        :param cluster: 
        :param attr_index: 
        :return: 
        '''
        p = {}
        for attr_index in attr_indexs:
            p[attr_index] = (self.gini_index(cluster, attr_index))
        attr_index = min(p, key=lambda x: p.get(x)[1])
        attr = p[attr_index][0]
        return  attr_index, attr

    def build_tree(self, cluster, attr_indexs):
        '''
        递归构建决策树
        :param cluster:给定数据集 
        :param attr_indexs: 给定的可供切分的特征编号的集合
        :return: 一个决策树结点
        '''
        flag = cluster[0, -1]
        for i in cluster[:, -1]:
            if i != flag:
                break
        else:
            return TreeNode(label=flag)
        if not attr_indexs:
            p = defaultdict(int)
            for line in cluster:
                p[line[-1]] += 1
            return TreeNode(label=max(p, key=p.get))
        for i in attr_indexs:
            flag = cluster[i][0]
            f = False
            for j in cluster[:, i]:
                if j != flag:
                    f = True
                    break
            if f:
                break
        else:
            p = defaultdict(int)
            for line in cluster:
                p[line[-1]] += 1
            return TreeNode(label=max(p, key=p.get))

        attr_index, attr = self.get_best_index(cluster, attr_indexs)
        left, right = self.devide_set(cluster, attr_index, attr)

        new_attr_indexs = attr_indexs - set([attr_indexs])
        left_branch = self.build_tree(left, new_attr_indexs)
        right_branch = self.build_tree(right, new_attr_indexs)

        return TreeNode(left_child=left_branch,
                        right_child=right_branch,
                        attr_index=attr_index,
                        attr=attr)

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集合X
        :param train_y: 训练集合Y（target）
        :return: None
        拟合决策树
        '''
        attr_indexs = set(range(train_x.shape[1]))
        self.train_x = np.c_[train_x, train_y]
        self.root = self.build_tree(self.train_x, attr_indexs)

    def predict_one(self, x):
        '''
        预测单个值
        :param x:  待预测的样本X
        :return: X所属的类别
        '''
        node_p = self.root
        while node_p.label == None:
            if x[node_p.attr_index] == node_p.attr:
                node_p = node_p.left_child
            else:
                node_p = node_p.right_child
        return node_p.label

    def CART(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集样本的类别集合
        预测多个值
        '''
        return np.array([self.predict_one(x) for x in test_x])

# 回归树


class RegressionTree:
    def load_data_set(self, file_name):
        '''
        # 加载数据
        :param file_name: 文件名称
        :return: 
        '''
        data_mat = []
        fr = open(file_name)
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            # 将里面的值映射成float,否则是字符串类型的
            flt_line = map(float, cur_line)
            data_mat.append(flt_line)
        return data_mat

    def bin_split_data_set(self, data_set, feature, value):
        '''
        # 按某列的特征值来划分数据集
        :param data_set: 输入的数据集合  
        :param feature: 特征列
        :param value: 切分点
        :return: 
        '''
        mat0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
        mat1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]
        return mat0, mat1

    def reg_leaf(self, data_set):
        '''
        将均值作为叶子节点
        :param data_set: 
        :return: 
        '''
        return np.mean(data_set[:, -1])

    def reg_err(self, data_set):
        '''
        计算误差 用来确定切分点
        :param data_set: 
        :return: 
        '''
        # 方差乘以行数
        return np.var(data_set[:, -1]) * np.shape(data_set)[0]

    def create_tree(self, data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
        feat, val = self.chooseBestSplit(data_set, leaf_type, err_type, ops)
        if feat == None:
            # 说明是叶节点，直接返回均值
            return val
        ret_tree = {}
        # 记录是用哪个特征作为划分
        ret_tree['spInd'] = feat
        # 记录是用哪个特征作为划分（以便于查找的时候，相等进入左树，不等进入右树）
        ret_tree['spVal'] = val
        # 按返回的特征来选择划分子集
        l_set, r_set = self.bin_split_data_set(data_set, feat, val)
        # 用划分的2个子集的左子集，递归建树
        ret_tree['left'] = self.create_tree(l_set, leaf_type, err_type, ops)
        ret_tree['right'] = self.create_tree(r_set, leaf_type, err_type, ops)
        return ret_tree

    def choose_best_split(self, data_set, leaf_type=reg_leaf, err_Type=reg_err, ops=(1, 4)):
        # 容许的误差下降值
        tol_s = ops[0]
        # 划分的最少样本数
        tol_n = ops[1]
        # 类标签的值都是一样的，说明没必要划分了，直接返回
        if len(set(data_set[:, -1].T.tolist()[0])) == 1:
            return None, leaf_type(data_set)
        # m是行数，n是列数
        m, n = np.shape(data_set)
        # 计算总体误差
        S = err_Type(self, data_set)
        # np.inf是无穷大的意思，因为我们要找出最小的误差值，如果将这个值设得太小，遍历时很容易会将这个值当成最小的误差值了
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        # 遍历每一个维度
        for featIndex in range(n-1):
            # 选出不同的特征值，进行划分,勘误：这里跟书上不一样，需修改
            for splitVal in set(data_set[:, featIndex].T.A.tolist()[0]):
                # 子集的划分
                mat0, mat1 = self.bin_split_data_set(data_set, featIndex, splitVal)
                # 划分的两个数据子集，只要有一个小于4，就说明没必要划分
                if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
                    continue
                    # 计算误差
                newS = err_Type(self, mat0) + err_Type(self, mat1)
                # 更新最小误差值
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
                    # 检查新切分能否降低误差
        if (S - bestS) < tol_s:
            return None, leaf_type(self, data_set)
        mat0, mat1 = self.bin_split_data_set(data_set, bestIndex, bestValue)
        # 检查是否需要划分（如果两个子集的任一方小于4则没必要划分）
        if (np.shape(mat0)[0] < tol_n) or(np.shape(mat1)[0] < tol_n):
            return None, leaf_type(self, data_set)
        return bestIndex, bestValue