import torch
from torch import nn
from torch.nn import functional as F

# 定义一个MLP的Module
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 1)
        self.modules_list = [self.hidden, self.relu, self.out]
        for idx, module in enumerate(self.modules_list):
            self.add_module(str(idx), module)
    
    def forward(self, X):
        for module in self.modules_list:
            X = module(X)
        return X
    
    def __getitem__(self, idx):
        return self.modules_list[idx]

X = torch.rand(2, 20)
net = MLP()
print(net[0].weight)
print(torch.__version__)