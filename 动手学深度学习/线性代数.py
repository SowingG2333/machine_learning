import torch

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