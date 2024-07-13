import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'])
print(type(net.state_dict()['2.bias']))

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet[0][1][0].bias.data)

# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

rgnet.apply(init_normal)
print(rgnet[0][0][0].weight.data)
print(rgnet[0][0][0].bias.data)

def init_const(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_const)
print(net[0].weight.data[0], net[0].bias.data[0])

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

net[0].apply(init_xavier)
print(net[0].weight.data[0], net[0].bias.data[0])

# 实现自定义初始化
def DIY_init(m):
    if type(m) == nn.Linear:
        print('init', *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(DIY_init)
net[0].weight[:2]
print(net[0].weight.data[0], net[0].bias.data[0])
