# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im

#dout：The Upstream Gradient (dout): How the final loss L changes with respect to this layer's output (out). This is ∂L/∂out. This value is passed in from the next layer.
#上层梯度（dout）：最终损失 L 随此层输出（out）的变化情况。这即为 ∂L/∂out 。该值由下一层传递过来
class Relu:
    def __init__(self):
        self.mask = None #初始化一个成员变量self.mask为None。mask用于记录输入中小于等于0的位置。
    #mask是由True/False构成的NumPy数组，它会把正向传播时的输入x的元素中小于等于0的地方保存为True，其他地方（大于0的元素）保存为False。
    def forward(self, x):
        self.mask = (x <= 0) #生成一个布尔数组，标记x中小于等于0的位置
        out = x.copy() #复制输入x，避免直接修改原数据
        out[self.mask] = 0 #将小于等于0的位置的值设为0，实现ReLU的功能

        return out

    def backward(self, dout):
        dout[self.mask] = 0 #反向传播时，输入中小于等于0的位置，梯度也设为0。
        dx = dout #将处理后的梯度赋值给dx。

        return dx #返回梯度


class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out
    def backward(self, dout):#上层梯度（dout）：最终损失 L 随此层输出（out）的变化情况。这即为 ∂L/∂out 。该值由下一层传递过来
        dx = dout * (1.0 - self.out) * self.out #据Sigmoid的导数公式，计算梯度。

        return dx


class Affine:
    def __init__(self, W, b):#创建一个仿射层对象时，需要告诉它“参数”是什么。对于仿射层，参数就是权重 W 和偏置b
        #将外部传入的权重和偏置矩阵保存为这个层的内部状态。
        self.W =W
        self.b = b
        
        self.x = None #为什么需要保存 x？因为在反向传播计算 dW 时，我们需要用到正向传播时的输入 x。
        self.original_x_shape = None #解决了输入数据可能是多维张量（如图像）的问题。
        # 权重和偏置参数的导数--存储反向传播计算出的梯度。计算出的梯度会被优化器（如SGD）用来更新 W 和 b。
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape #保存输入x的原始形状（方便反向传播还原）。
        x = x.reshape(x.shape[0], -1) #将输入x展平成二维（批量大小，特征数）。
        self.x = x

        out = np.dot(self.x, self.W) + self.b #Y=XW+B

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) #∂L/∂X = (∂L/∂Y) * (∂Y/∂X)，∂L/∂Y 就是输入的 dout，∂Y/∂X (Y对X的偏导) 是 W。
        self.dW = np.dot(self.x.T, dout) #∂L/∂W = (∂Y/∂W) * (∂L/∂Y)
        self.db = np.sum(dout, axis=0) #∂Y/∂B 是 1。所以 db = dout * 1 = dout，np.sum(dout, axis=0) 正是沿着批次维度（第0轴）进行求和，得到一个和 b 形状相同的梯度向量。
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx
# x = x.reshape(x.shape[0], -1)这行代码的作用是把输入 x 变成一个二维数组（矩阵），方便后续的矩阵运算。详细解释如下：

# x.shape[0]：表示输入 x 的第一个维度的大小，通常是“批量大小”（batch size），即一次输入多少个样本。
# -1：在 reshape 里，-1 表示“自动计算”这一维的大小，使得总元素个数不变。
# x.reshape(x.shape[0], -1)：把 x 变成形状为 (batch_size, 特征数) 的二维数组。
# 举例说明：

# 假设 x 是一个形状为 (10, 3, 28, 28) 的张量，表示10张3通道28x28的图片。

# x.shape[0] 是 10。
# x.reshape(10, -1) 会把每张图片展平成一行，总共 3×28×28=2352 个特征，所以新形状是 (10, 2352)。
# 这样做的目的是：
# 无论输入是几维的（比如图片、序列等），都能统一变成“批量 × 特征数”的二维形式，方便和权重矩阵做矩阵乘法。

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:#标签t是类别索引，先把y中对应正确类别的位置减1--这样就实现了(y-t)的效果，再除以batch_size
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv层的情况下为4维，全连接层的情况下为2维  

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
