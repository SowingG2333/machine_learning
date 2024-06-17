import torch
import torch.distributions as multionmial
from d2l import torch as d2l

# # 输出一系列的投掷骰子结果
# fair_probs = torch.ones([6]) / 6
# result = multionmial.Multinomial(1000, fair_probs).sample()
# print(result)
# result = result / 1000
# print(result)

# 显示概率随次数增加而收敛的图像
fair_probs = torch.ones([6]) / 6
counts = multionmial.Multinomial(100, fair_probs).sample((500,))
print(counts)
cum_counts = counts.cumsum(dim=0)
print(cum_counts)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)
print(estimates)

# 绘图
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()