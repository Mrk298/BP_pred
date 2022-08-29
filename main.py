
#####对山东某地区历史温度数据进行BP神经网络训练，通过前三小时温度数据，预测第四小时温度值。
import pandas as pd
import numpy as np
import random
import time
import math

############################### 定义神经网络的参数####################################
#选用典型的三层神经网路（输入层+一层隐含层+输出层），节点数设置为3-7-1 ：

d = 3  # 输入节点个数
l = 1  # 输出节点个数
q = 2 * d + 1  # 隐层个数,采用经验公式2d+1
eta = 0.5  # 学习率
error = 0.002  # 精度
train_num = 480  # 训练数据个数
test_num = 240  # 测试数据个数

# #############################　初始化权值阈值　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
# 输入层到隐含层的权值为w1,矩阵大小为d*q；隐含层到输出层的权值为w2，矩阵大小为q*l；隐含层阈值为b1，输出层阈值为b2。

w1 = [[random.random() for i in range(q)] for j in range(d)]  #3*7
w2 = [[random.random() for i in range(l)] for j in range(q)]  #7*1
b1 = [random.random() for i in range(q)] #1*7
b2 = [random.random() for i in range(l)] #1*1

# ＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃　读取气温数据　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
# 通过csv文件读取数据，并将网络的输入输出数据分组。csv文件数据排列为30*24行一列。

dataset = pd.read_csv('tem.csv', delimiter=",")
dataset = np.array(dataset)
m, n = np.shape(dataset) #获得数据的维度
totalX = np.zeros((m - d, d))
totalY = np.zeros((m - d, l))
for i in range(m - d):  # 分组：前三个值输入，第四个值输出
    totalX[i][0] = dataset[i][0]
    totalX[i][1] = dataset[i + 1][0]
    totalX[i][2] = dataset[i + 2][0]
    totalY[i][0] = dataset[i + 3][0]

#　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃ 　归一化数据　　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
# 将数据归一化到0-1内，并截取1-20日数据放到训练集，21-30日数据放到测试集。



Normal_totalX = np.zeros((m - d, d))
Normal_totalY = np.zeros((m - d, l))
nummin = np.min(dataset)
nummax = np.max(dataset)
dif = nummax - nummin
for i in range(m - d):
    for j in range(d):
        Normal_totalX[i][j] = (totalX[i][j] - nummin) / dif
    Normal_totalY[i][0] = (totalY[i][0] - nummin) / dif

# ＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃　截取训练数据　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
# 截取划分训练测试数据

trainX = Normal_totalX[:train_num - d, :]  # 训练数据
trainY = Normal_totalY[:train_num - d, :]
testX = Normal_totalX[train_num:, :]  # 测试数据
testY = Normal_totalY[train_num:, :]
m, n = np.shape(trainX)
print(m)  #m=477
# ＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃　实现sigmoid函数　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃

def sigmoid(iX):
    for i in range(len(iX)):
        iX[i] = 1 / (1 + math.exp(-iX[i]))
    return iX


# ＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃　网络训练　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
# 采用梯度下降法
start = time.perf_counter()  # 起始时间
iter = 0
while True:
    sumE = 0
    for i in range(m):  # 每行循环
        alpha = np.dot(trainX[i], w1)   #1*3 点积 3*7 得 1*7
        b = sigmoid(alpha - b1)     #1*7 减去 1*7 得 1*7
        beta = np.dot(b, w2)
        predictY = sigmoid(beta - b2)
        E = (predictY - trainY[i]) * (predictY - trainY[i])
        sumE += E
        # 梯度下降法  修改权值参数
        g = predictY * (1 - predictY) * (trainY[i] - predictY)
        e = b * (1 - b) * ((np.dot(w2, g.T)).T)
        w2 += eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))
        b2 -= eta * g
        w1 += eta * np.dot(trainX[i].reshape((d, 1)), e.reshape((1, q)))
        b1 -= eta * e
    sumE = sumE / m
    iter += 1
    if iter % 10 == 0:  # 每训练10次，输出误差
        print("第 %d 次训练后,误差为：%g" % (iter, sumE))
    if sumE < error:  # 误差小于0.002，退出循环
        break
print("循环训练总次数：", iter)

end = time.perf_counter()  # 结束时间
print("运行耗时(s)：", end - start)
print()
print("输出权值参数的值：")
print("w1的值是：",w1)
print("b1的值是：",b1)
print("w2的值是：",w2)
print("b2的值是：",b2)
print()

# ＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃　测试,求均方根误差　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
# 求取测试集均方误差。
m, n = np.shape(testX)
MSE = 0
for i in range(m):
    alpha = np.dot(testX[i], w1)
    b = sigmoid(alpha - b1)
    beta = np.dot(b, w2)
    y = sigmoid(beta - b2)
    testY[i] = testY[i] * dif + nummin  # 反归一化
    y = y * dif + nummin
    MSE = (y - testY[i]) * (y - testY[i]) + MSE
MSE = MSE / m
print("测试集均方误差：", MSE)
print()

# ＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃　预测　＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
#最后简单实现下预测。采用5月1日连续三小时温度，预测第4小时温度。

def predict(iX):
    iX = (iX - nummin) / dif  # 归一化
    alpha = np.dot(iX, w1)
    b = sigmoid(alpha - b1)
    beta = np.dot(b, w2)
    predictY = sigmoid(beta - b2)
    predictY = predictY * dif + nummin  # 反归一化
    return predictY


XX = [18.3, 17.4, 16.7]
XX = np.array(XX)
print("[18.3,17.4,16.7]输入下,预测气温为：", predict(XX))