import torch
import numpy as np

# 实例化标量并进行运算
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y)
print(x * y)
print(x / y)
print(x ** y)

# 实例化一个张量
x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

# 实例化一个矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

# 对称矩阵与转置
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)

# 高维张量
X = np.arange(24).reshape(2, 3, 4)
print(X)

# 张量的基本运算
A = torch.arange(20).reshape(5, 4)
B = A.clone() # 分配新内存，将A赋值给B
print(A)
print(A + B)
print(A * B)
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)

# 张量的降维
x = torch.arange(4, dtype=torch.float32)
print(x.sum())
print(A.sum()) # 使用求和函数降维（但是会降成0维标量）
A_sum_axis0 = A.sum(axis = 0)
print(A, A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis = 1)
print(A, A_sum_axis1, A_sum_axis1.shape)
print(A.sum(axis = [0, 1]) == A.sum())
print(A.float().mean(axis = 0)) # 求平均值（沿列方向）
print(A.numel()) # 求元素总个数
print(A, A.float().mean(axis = 0))
print(A, A.sum(axis = 0) / A.shape[0])

# 非降维求和
sum_A = A.sum(axis = 1, keepdim=True)
print(A_sum_axis1, sum_A)
print(A / sum_A) # 由于sum_A的形状为[5, 1]，而A的形状是[5, 4]，维度相同，最后一维维数个个数为1，可运用广播机制
print(A, A.cumsum(axis=0)) # 不会降低维度

# 点积运算
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))

# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A.float(), B))

# L2范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# L1范数
print(torch.abs(u).sum())

# LP范数
print(torch.norm(torch.ones((4, 9))))