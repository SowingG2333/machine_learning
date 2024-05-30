import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_inline import backend_inline
from d2l import torch as d2l

# 定义一个数学函数
def f(x):
    return 3 * x ** 2 - 4 * x

# 定义近似求导函数
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

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
    
# 画图
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

# 作业1
x = np.arange(0, 2, 0.01)
plot(x, [x ** 3 - x ** (-1), 2 * x - 2], 'x', 'f(x)', legend=["f(x)", '切线(x = 1)'])

# 作业2：[6 * x1, 5 * e ** x2]

# 作业3：对 (x ** 2 + y **2) ** (1 / 2) 分别对x和y求偏导

# 作业4 ：略
