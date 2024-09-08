import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    pool_height, pool_width = pool_size
    Y = torch.zeros(X.shape[0] - pool_height + 1, X.shape[1] - pool_width + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+pool_height, j:j+pool_width].max()
            if mode == 'avg':
                Y[i, j] = X[i:i+pool_height, j:j+pool_width].mean()
    return Y

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
nn_pool_2d_layer = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
Y = nn_pool_2d_layer(X)
print(Y)

X = torch.cat((X, X + 1), 1)
nn_pool_2d_layer = nn.MaxPool2d(3, padding=1, stride=2)
print(nn_pool_2d_layer(X))