import torch
import torch.nn.functional as F
from torch import nn

# 定义一个不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, X):
        return X - X.mean()
    
# 合并到sequential模块
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
net = net.double()

Y = net(torch.rand((4, 8), dtype=torch.float64))
print(Y.mean())

# 定义一个带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, out_units, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = torch.randn(in_units, out_units)
        self.bias = torch.randn(out_units)
    def forward(self, X):
        linear = torch.matmul(X, self.weights) + self.bias
        return F.relu(linear)
    
linear = MyLinear(5, 3)
print(linear.weights)

print(linear(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))