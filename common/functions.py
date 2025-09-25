# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0) #每一列都减去该列最大值，防止指数运算时数值溢出（比如exp(1000)会溢出）。
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

#y：模型的输出（通常是softmax的概率），可以是一维或二维数组。
#t：真实标签，可以是one-hot向量或类别索引。
def cross_entropy_error(y, t):
    #如果y是一维（单个样本），则把y和t都变成二维（批量为1），方便后续统一处理。
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)#转换后的 y 形状： (1, 3)
        
    # 如果t和y的元素个数相同，说明t是one-hot向量。用argmax把one-hot向量转换为类别索引。
    if t.size == y.size:
        t = t.argmax(axis=1)
    #原始 t：
#  [[0 1 0]
#   [0 0 1]
#   [1 0 0]]
# 转换后的 t： [1 2 0]

    #y[np.arange(batch_size), t]：取出每个样本预测的正确类别的概率。       
    # np.log(... + 1e-7)：对概率取对数，加1e-7防止log(0)。
    #-np.sum(...) / batch_size：对所有样本的损失求平均，得到交叉熵损失。
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#举例
# 假设有两个样本，3个类别：

# y = [[0.1, 0.7, 0.2], [0.3, 0.2, 0.5]] # softmax输出
# t = [1, 2] # 正确类别索引
# 则损失为：

# 第一行取0.7，第二行取0.5，分别取log后求和取负，再除以2。

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
