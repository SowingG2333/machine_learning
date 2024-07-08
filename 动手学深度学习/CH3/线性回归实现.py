import random
import torch
from d2l import torch as d2l

# 噪声生成函数
def synth_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples,len(w)))
    Y = torch.matmul(X, w) + b
    Y += torch.normal(0, 0.01, Y.shape)
    return X, Y.reshape((-1, 1))

# 人工构造包括噪声的数据集
real_w = torch.tensor([2, -3.4])
real_b = 4.2
features, labels = synth_data(real_w, real_b, 1000)

# 绘制散点图
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()

# 定义数据集局部读取函数
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, Y in data_iter(batch_size, features, labels):
    print(X, '\n', Y)
    break

# 初始化模型参数（自动微分）
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义线性回归模型
def linear_model(X, w, b):
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义随机梯度函数
def SGD_fuc(params, learn_rt, batch_size):
    with torch.no_grad():
        for param in params:
            param -= learn_rt * param.grad / batch_size
            param.grad.zero_()

# 开始训练
learn_rt = 0.02
num_epochs = 10
net = linear_model
loss = squared_loss

for epoch in range(num_epochs):
    for X, Y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), Y)
        l.sum().backward()
        SGD_fuc([w, b], learn_rt, batch_size)
    with torch.no_grad():
        trained_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(trained_l.mean()):f}')
