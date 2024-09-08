import torch
from torch import nn
from torch.nn import functional as F

# 定义一个MLP的Module
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 1)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
X = torch.rand(2, 20)
net = MLP()
print(net(X))

# 手动实现sequential类
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for index, module in enumerate(args):
            self._modules[str(index)] = module
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256, 1024))
print(net(X))
    
# 实现灵活定义的块
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
net(X)

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)

# 实现平行块
class ParallelModule(nn.Module):
    def __init__(self, input, output,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net1 = nn.Linear(input, output)
        self.net2 = nn.Linear(input, output)

    def forward(self, X):
        out1 = self.net1(X)
        out2 = self.net2(X)
        return out1 + out2
    
# 实现同一网络的多个实例
def multi_instantiation (input, output, num_instation):
    layer = []
    for i in range(num_instation):
        X = ParallelModule(input, output)
        layer.append(X)
    return MySequential(*layer)