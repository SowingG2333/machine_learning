import torch
from d2l import torch as d2l

x = torch.arange(-10.0, 10.0, 0.01, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), x.grad, 'x', 'relu(x)', figsize=(5, 2.5))
d2l.plt.show()

y.sum().backward()
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l.plt.show()

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.show()

x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l.plt.show()


