# coding: utf-8
import numpy as np

class SGD:

    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        # 学习率（learning rate），控制参数更新的步长
        self.lr = lr
        
    def update(self, params, grads):
        # 对每个参数进行更新
        for key in params.keys():
            # 参数更新公式：params = params - lr * grads
            # 沿着梯度的反方向更新参数，以最小化损失函数
            params[key] -= self.lr * grads[key] 


class Momentum:

    """Momentum SGD（动量随机梯度下降法）"""

    def __init__(self, lr=0.01, momentum=0.9):
        # 学习率（learning rate），控制参数更新的步长
        self.lr = lr
        # 动量系数（momentum），通常设为0.9，用于加速收敛
        self.momentum = momentum
        # 速度向量（velocity），用于积累梯度信息，初始为None
        self.v = None
        
    def update(self, params, grads):
        # 如果速度向量未初始化，则创建与参数相同形状的零向量
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        # 对每个参数进行更新
        for key in params.keys():
            # 更新速度向量：v = momentum * v - lr * grads
            # 动量项帮助梯度在平坦区域加速，在陡峭区域减速
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            # 更新参数：params += v
            params[key] += self.v[key]


class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:

    """AdaGrad（自适应梯度算法）"""

    def __init__(self, lr=0.01):
        # 学习率（learning rate），控制参数更新的步长
        self.lr = lr
        # 历史梯度平方和（historical gradient squared），用于自适应调整学习率
        self.h = None
        
    def update(self, params, grads):
        # 如果历史梯度平方和未初始化，则创建与参数相同形状的零向量
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        # 对每个参数进行更新
        for key in params.keys():
            # 累积历史梯度平方和：h += grads^2
            self.h[key] += grads[key] * grads[key]
            # 参数更新公式：params -= lr * grads / sqrt(h + epsilon)
            # 自适应学习率：频繁更新的参数学习率降低，稀疏参数学习率保持较高
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop（均方根传播算法）"""

    def __init__(self, lr=0.01, decay_rate=0.99):
        # 学习率（learning rate），控制参数更新的步长
        self.lr = lr
        # 衰减率（decay rate），通常设为0.99，用于指数移动平均
        self.decay_rate = decay_rate
        # 历史梯度平方和的指数移动平均（exponential moving average of squared gradients）
        self.h = None
        
    def update(self, params, grads):
        # 如果历史梯度平方和未初始化，则创建与参数相同形状的零向量
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        # 对每个参数进行更新
        for key in params.keys():
            # 更新指数移动平均：h = decay_rate * h + (1 - decay_rate) * grads^2
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            # 参数更新公式：params -= lr * grads / sqrt(h + epsilon)
            # 自适应学习率：使用指数移动平均避免学习率过快下降
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """Adam (Adaptive Moment Estimation) - 自适应矩估计优化器"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        # 学习率（learning rate），控制参数更新的步长
        self.lr = lr
        # 一阶矩的衰减率（beta1），通常设为0.9，用于梯度的指数移动平均
        self.beta1 = beta1
        # 二阶矩的衰减率（beta2），通常设为0.999，用于梯度平方的指数移动平均
        self.beta2 = beta2
        # 迭代次数（iteration count），用于偏差修正
        self.iter = 0
        # 一阶矩（first moment），梯度的指数移动平均
        self.m = None
        # 二阶矩（second moment），梯度平方的指数移动平均
        self.v = None
        
    def update(self, params, grads):
        # 如果一阶矩和二阶矩未初始化，则创建与参数相同形状的零向量
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        # 迭代次数加1
        self.iter += 1
        # 计算偏差修正的学习率：lr_t = lr * sqrt(1 - beta2^iter) / (1 - beta1^iter)
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        # 对每个参数进行更新
        for key in params.keys():
            # 更新一阶矩：m = beta1 * m + (1 - beta1) * grads
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # 更新二阶矩：v = beta2 * v + (1 - beta2) * grads^2
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            # 另一种等价形式：m += (1 - beta1) * (grads - m)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            # v += (1 - beta2) * (grads^2 - v)
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            # 参数更新公式：params -= lr_t * m / sqrt(v + epsilon)
            # 结合动量和自适应学习率，偏差修正后更新参数
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            # 注释掉的代码是另一种偏差修正方式（已实现）
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
