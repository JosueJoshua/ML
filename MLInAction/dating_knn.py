# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 数据量
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 测试数据与所有数据的差值
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 按行相加。0为按列相加
    distances = sqDistances ** 0.5  # 当前测试数据与各个数据间的欧式距离
    sortedDistIndicies = distances.argsort()  # 从小到大排序后的索引值
    classCount = {}
    for i in range(k):
        # label 值
        voteIlabel = labels[sortedDistIndicies[i]]  # 选取k 个距离最小的数据的 label。
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算对应label 出现的次数
    # 按照value 值降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print('classCount', classCount)
    # print('sortedClassCount', sortedClassCount)
    # print(sortedClassCount[0], '+', sortedClassCount[0][0])
    return sortedClassCount[0][0]  # 预测 label 值

def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 文件行数
    returnMat = np.zeros([numberOfLines, 3])  # 特征值集合
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 数据处理
        line = line.strip()
        listFromLine = line.split('\t')
        # print('listFromLine', listFromLine)
        returnMat[index, :] = listFromLine[0:3]  # 所有人的特征集合
        if listFromLine[-1].isdigit():
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector  # 特征集合，类别集合

def paintChart():
    datingDataMat, datingLabels = file2matrix('DataSet/KNN/datingTestSet.txt')
    print(datingDataMat)
    print(datingLabels[:20])

    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure()
    # ax = fig.add_subplot(211)  # 将画布分割成1行1列，图像画在从左到右从上到下的第一块
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
    #            15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    # plt.xlabel('玩视频游戏所耗时间百分比')
    # plt.ylabel('每周消费的冰淇淋公升数')
    # plt.grid(True)  # 添加网格

    # ax1 = fig.add_subplot(111)  # 将画布分割成1行1列，图像画在从左到右从上到下的第一块
    axes = plt.subplot(111)
    type1_x, type1_y, type2_x, type2_y, type3_x, type3_y = [], [], [], [], [], []
    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:
            type1_x.append(datingDataMat[i][0])
            type1_y.append(datingDataMat[i][1])

        if datingLabels[i] == 2:
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    print('type1_x:', type1_x)
    type1 = axes.scatter(type1_x, type1_y, s=20, c='r')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='b')
    type3 = axes.scatter(type3_x, type3_y, s=60, c='k')

    plt.legend((type1, type2, type3), ('不喜欢', '魅力一般', '极具魅力'), loc='upper left')
    plt.xlabel('每年获取的飞行常客里程数')
    plt.ylabel('玩视频游戏所耗时间百分比')
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 参数 0 使得函数可以从列中选取最小值，而不是选取当前行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals  # 数据整体范围量
    normDataSet = np.zeros(shape(dataSet))  # 全为 0 的array
    m = dataSet.shape[0]  # 数据量
    ## 归一化
    normDataSet = dataSet - tile(minVals, (m, 1))  # 所有数据在整体范围量中所占大小
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 归一化
    return normDataSet, ranges, minVals  # 归一化后的数据，数据整体范维量，最小数据值

def datingClassTest():
    hoRatio = 0.10  # 数据占比比例
    datingDataMat, datingLabels = file2matrix('DataSet/KNN/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 数据量
    numTestVecs = int(m * hoRatio)  # 作为输入数据的数量(即预测数据量)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d'
              % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))
# datingClassTest()
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per years?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('DataSet/KNN/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals), normMat, datingLabels, 3)
    print('You will probably like this person: ', resultList[classifierResult-1])

classifyPerson()