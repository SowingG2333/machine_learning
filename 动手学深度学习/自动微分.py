import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_inline import backend_inline
from d2l import torch as d2l
import math

x = torch.arange(4.0)
print(x)

# 跟踪x的梯度计算
x.requires_grad_(True)
print(x.grad)# 默认为None

# 利用反向传播函数计算y关于x分量的梯度
y = 2 * torch.dot(x, x)
y.backward()
print(x.grad)

# 验证梯度计算与理论计算是否相等
print(x.grad == 4 * x)

# 清除积累的梯度，重新进行梯度计算
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 探究非标量y的梯度计算
x.grad.zero_()
y = x * x
y.sum().backward()# 此处使用sum()降维
print(x.grad)

# 采用不降维的方法
x.grad.zero_()
y = x * x
y.backward(torch.ones_like(x))
print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)# 此时反向传播是将u视为一个常数，而不是x * x
# 已经记录结果后再进行反向传播
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x

# 进行python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(3, 3), requires_grad=True)
d = f(a)
d.backward(torch.ones_like(a))
print(a.grad == d / a)

# 作业1：因为求二阶导数的运算步骤比一阶复杂许多，同时需要更大的储存空间
# 作业2：在此之前我已经触发过，会提示错误：在对一个计算图调用.backward()后会清空中间结果，所以如果企图再次对同一个计算图进行反向传播就会触发错误，除非设置retain_graph=True（在RNN中会用到）
# 作业3：会触发非标量函数的错误，需要设置torch.ones_like(a)
# 作业4：
def d(a):
    return a ** 3
m = torch.randn(size=(3, 3), requires_grad=True)
n = d(m).sum()
n.backward()
print(m.grad == 3 * (m ** 2))
# 作业5：
# 定义一个绘图函数，使用svg格式
def use_svg_display(): 
    backend_inline.set_matplotlib_formats('svg')

# 初始化图表大小
def set_figsize(figsize=(3.5, 2.5)): 
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
    
# 初始化图表axis
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
# 定义绘图函数
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    # 绘制数据点
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    

    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()
    matplotlib.use('TkAgg')

x = torch.arange(0, 2 * math.pi, 0.1, requires_grad=True)
y = torch.sin(x)
y.sum().backward()
x.grad
# 使用 .detach() 保存数据作图
x_np = x.detach().numpy()
y_np = y.detach().numpy()
grad_np = x.grad.detach().numpy()
plot(x_np, [y_np, grad_np], 'x', 'f(x)', legend=['f(x)', 'df(x) / dx'])

