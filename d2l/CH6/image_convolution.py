import torch
from torch import nn
from d2l import torch as d2l

# 定义卷积操作函数
def corr2d(X, K):
    h, w = K.shape
    r = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r[i, j] = (X[i:i + h, j:j + w] * K).sum() 
    return r

# 定义卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

# 检测图像边缘
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)

lr = 3e-2  # 学习率

module = Conv2D(kernel_size=(1, 2))

for i in range(20):
    Y_hat = module(X)
    l = (Y - Y_hat) ** 2
    module.zero_grad()
    l.sum().backward()
    module.weight.data -= lr * module.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(module.weight.data.reshape((1, 2)))

# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(20):
    Y_hat = conv2d(X)
    l = (Y - Y_hat) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))