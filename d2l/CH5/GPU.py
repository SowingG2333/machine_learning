import torch
from torch import nn

print(torch.cuda.device_count())

x = torch.tensor([1, 2, 3])
print(x.device)

X = torch.ones(2, 3, device=torch.device('cuda'))
print(X)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=torch.device('cuda'))
print(net(X))
print(net[0].weight.data.device)