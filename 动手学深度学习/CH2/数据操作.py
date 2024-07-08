# 导入所需库
import torch

# 创建一个0到11，有十二个元素的的张量
x = torch.arange(12)
print(x)
# 打印x的形状，x的元素个数
print(x.shape)
print(x.numel())

# 创建一个新的张量X，形状为三行四列
X = x.reshape(-1, 4)
print(X)

# 创建一个形状为（2，3，4）的三维全零张量y
y = torch.zeros((2, 3, 4))
print(y)

# 创建一个形状为（2，3，4）的三维全一张量z
z = torch.ones((2, 3, 4))
print(z)

# 创建一个三行四列的随机张量（每个元素都是从均值为0，标准差为1的正态分布中随机抽取的）并赋值给random_matrix
random_matrix = torch.randn(3, 4)
print(random_matrix)

# 创建一个[1, 2, 3]的pytorch张量并赋给tensor_matrix
tensor_matrix = torch.tensor([1, 2, 3])
print(tensor_matrix)

# 创建两个pytorch张量赋给x和y并对其进行加减，乘方和e的幂运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x * y)
print(x ** y)
print(torch.exp(x))

# 创建一个0到11的一维张量并且将其形状改变为三行四列
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("X = " + str(X))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 按照第零个维度（行）将X与Y张量进行连接
print(torch.cat((X, Y), dim = 0))

# 按照第一个维度（列）将X与Y张量进行连接
print(torch.cat((X, Y), dim = 1))

# 对张量元素进行逻辑运算
print(X < Y)
print(X > Y)
print(X == Y)
print(X.sum())

'''
探究张量的广播机制：
如果张量的维度不相同 那么会在较小维度的张量前面补一个维度 该维度元素个数为1 直到两个张量的维度相同
（注意：补全到维度相同时两个张量最后一个维度元素个数必须相同或者其中一个为1）
如果两个张量在某个维度上的大小不同 且其中一个张量在该维度上的大小为1 那么会将这个维度的大小扩展到与另一个张量在该维度上的大小相同。
'''
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
X = torch.arange(12, dtype= torch.float32).reshape(3, -1)
print(a)
print(b)
print(a + b)

tensor_1 = torch.arange(24).reshape((2, 4, 3))
tensor_2 = torch.arange(6).reshape((2, 1, 3))
print(tensor_1)
print(tensor_2)
print(tensor_1 + tensor_2)

# 打印X的最后一行
print(X[-1])
# 打印X的第二行和第三行（前闭后开）
print(X[1: 3])
# 将X第一行和第二行的元素都设置为12
X[0:2, :] = 12
print(X)

# 探究数据处理的内存优化
before = id(Y)
Y = Y + X
print(id(Y) == before)

