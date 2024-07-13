import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net[0].weight)

X = torch.rand(2, 20)
net(X)

print(net[0].weight.shape, net[2].weight.shape)