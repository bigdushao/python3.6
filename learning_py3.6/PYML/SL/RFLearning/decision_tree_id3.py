# coding:utf-8
'''
使用id3构建决策树
ID3算法通过计算每个属性的信息增益，认为信息增益高的是好属性，每次划分选取信息增益高的属性为划分标准，重复这个过程，
直至生成一个能完美分类训练样列的决策树
两个计算信息熵和信息增益

1）数据准备：需要对数值型数据进行离散化 
2）ID3算法构建决策树：
    如果数据集类别完全相同，则停止划分
    否则，继续划分决策树： 
        计算信息熵和信息增益来选择最好的数据集划分方法；
        划分数据集
        创建分支节点：
        对每个分支进行判定是否类别相同，如果相同停止划分，不同按照上述方法进行划分。
'''
from math import log
import operator

def createDataSet():
    '''
    创建测试数据集
    :return: 
    '''
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calc_shannon_ent(dataSet):
    '''
    计算信息熵
    :param dataSet: 
    :return: 
    '''
    # nrows
    num_entries = len(dataSet)
    #为所有的分类类目创建字典
    labelCounts = {}
    for featVec in dataSet:
        currentLable = featVec[-1] #取得最后一列数据
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1
    #计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / num_entries
        shannonEnt -= prob * log(prob, 2) # 以2为底的对数的求解
    return shannonEnt

#定义按照某个特征进行划分的函数splitDataSet
#输入三个变量（待划分的数据集，特征，分类值）
def splitDataSet(dataSet, axis, value):
    retDataSet=[]
    for featVec in dataSet: # featVec = [1, 1, 'yes']
        if featVec[axis] == value: # featVec[axis] = 1
            reduceFeatVec = featVec[:axis] # featVec[:axis] = []
            reduceFeatVec.extend(featVec[axis+1:]) # a = [1, 2, 3] ; b = [4, 5, 6] a.extend(b) = [1, 2, 3, 4, 5, 6]
            retDataSet.append(reduceFeatVec)# a = [1, 2, 3]; b = [4, 5, 6] a.append(b) = [[1, 2, 3], [4, 5, 6]]
    return retDataSet #返回不含划分特征的子集

#定义按照最大信息增益划分数据的函数
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1 # 获取地一个list的第一行计算元素的个数减去标签值
    baseEntropy = calc_shannon_ent(dataSet) # 香农熵 对于一个数据集的信息熵
    bestInforGain = 0
    bestFeature = -1
    for i in range(numFeature):
        featList = [number[i] for number in dataSet] # 得到某个特征下所有值（某列）
        uniqualVals = set(featList) # set无重复的属性特征值
        newEntropy = 0
        # 循环计算每个列的信息熵
        for value in uniqualVals:
            subDataSet = splitDataSet(dataSet, i, value) # 按照某个特正将数据切分后的数据集
            prob = len(subDataSet) / float(len(dataSet)) # 即p(t) len(subDataSet) 是按照某列进行数据分割后的数据的集合， len(dataSet)原始数据的数据集合
            newEntropy += prob * calc_shannon_ent(subDataSet) # 对各子集香农熵求和 p(i) * H(i)
        infoGain = baseEntropy - newEntropy # 计算信息增益
        #最大信息增益 比较赋值的算法，例如: if a > b
        if infoGain > bestInforGain:
            bestInforGain = infoGain
            bestFeature = i
    return bestFeature # 返回特征值

# 投票表决代码
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        # 将dict根据value倒序排序，将最大value的值获取。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    # 标签值 yes or no 单独取出最后一个字段,取出最后一列的值.
    classList = [example[-1] for example in dataSet]
    #类别相同，停止划分 即所有的分类标签都相同
    if classList.count(classList[-1]) == len(classList): # 计算列表中的某个元素的个数。例如:[1, 2, 2, 3, 3, 3].count(1) == 1;[1, 2, 2, 3, 3, 3].count(2) == 2
        return classList[-1]
    #长度为1，返回出现次数最多的类别
    if len(classList[0]) == 1:
        return majorityCnt(classList)

    #按照信息增益最高选取分类特征属性
    bestFeat = chooseBestFeatureToSplit(dataSet)#返回分类的特征序号
    bestFeatLable = labels[bestFeat] #该特征的label
    myTree = {bestFeatLable:{}} #构建树的字典
    del(labels[bestFeat]) #从labels的list中删除该label
    # 得到分类的特征值的集合，[0, 1, 1, 0, 1]
    featValues = [example[bestFeat] for example in dataSet]# 得到列表包含的所有属性值 例如:选择了第一列作为bestFeat，那么featValue的值为0,1
    uniqueVals = set(featValues) # 将所有的分类特征值进行去重 Set(0, 1)
    for value in uniqueVals: # 根据这个value,下标， dataSet进行splitDataSet并将结果赋值给 字典
        # 为了保证每次调用函数 createTree() 时不改变原始列表类型，使用新变量 subLabels 代替原始列表
        subLabels = labels[:] # 这行代码复制了类标签，并将其存储在新列表变量 subLabels 中
        #构建数据的子集合，并进行递归
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#输入三个变量（决策树，属性特征标签，测试的数据）
def classify(inputTree,featLables,testVec):
    firstStr = list(inputTree.keys())[0] #获取树的第一个特征属性
    secondDict = inputTree[firstStr] #树的分支，子集合Dict
    featIndex = featLables.index(firstStr) #获取决策树第一层在featLables中的位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

major_result = majorityCnt([1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
print(major_result)