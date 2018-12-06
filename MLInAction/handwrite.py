# -*- coding: utf-8 -*-
import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 数据量
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 测试数据与所有数据的差值
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

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect  # 一条数据

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('DataSet/KNN/digits/trainingDigits')
    m = len(trainingFileList)  # 数据量
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 文件名
        fileStr = fileNameStr.split('.')[0]  # 文件名前缀
        classNumStr = int(fileStr.split('_')[0])  # 从文件名前缀中取label 值
        hwLabels.append(classNumStr)  # label 集合
        trainingMat[i, :] = img2vector('DataSet/KNN/digits/trainingDigits/%s' % fileNameStr)  # 数据集合
    testFileList = listdir('DataSet/KNN/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    testMat = np.zeros((m, 1024))
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('DataSet/KNN/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  # 预测 label
        print('the classifier came back with: %d, the real answer is: %d'
              % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print('\nthe toal number of errors is: %d' % errorCount)
    print('\nthe total error rate is: %f' % (errorCount/float(mTest)))

handwritingClassTest()