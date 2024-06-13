import matplotlib.pyplot as plt

# 数据
data = """
Random state: 0, Common Neighbor accuracy: 0.14, Katz accuracy: 0.64
Random state: 1, Common Neighbor accuracy: 0.12, Katz accuracy: 0.64
Random state: 2, Common Neighbor accuracy: 0.14, Katz accuracy: 0.59
Random state: 3, Common Neighbor accuracy: 0.13, Katz accuracy: 0.57
Random state: 4, Common Neighbor accuracy: 0.12, Katz accuracy: 0.54
Random state: 5, Common Neighbor accuracy: 0.22, Katz accuracy: 0.76
Random state: 6, Common Neighbor accuracy: 0.12, Katz accuracy: 0.72
Random state: 7, Common Neighbor accuracy: 0.13, Katz accuracy: 0.58
Random state: 8, Common Neighbor accuracy: 0.1, Katz accuracy: 0.58
Random state: 9, Common Neighbor accuracy: 0.17, Katz accuracy: 0.6
Random state: 10, Common Neighbor accuracy: 0.16, Katz accuracy: 0.7
Random state: 11, Common Neighbor accuracy: 0.1, Katz accuracy: 0.67
Random state: 12, Common Neighbor accuracy: 0.13, Katz accuracy: 0.6
Random state: 13, Common Neighbor accuracy: 0.07, Katz accuracy: 0.54
Random state: 14, Common Neighbor accuracy: 0.16, Katz accuracy: 0.75
Random state: 15, Common Neighbor accuracy: 0.11, Katz accuracy: 0.48
Random state: 16, Common Neighbor accuracy: 0.17, Katz accuracy: 0.65
Random state: 17, Common Neighbor accuracy: 0.06, Katz accuracy: 0.63
Random state: 18, Common Neighbor accuracy: 0.13, Katz accuracy: 0.52
Random state: 19, Common Neighbor accuracy: 0.14, Katz accuracy: 0.71
Random state: 20, Common Neighbor accuracy: 0.07, Katz accuracy: 0.66
Random state: 21, Common Neighbor accuracy: 0.13, Katz accuracy: 0.65
Random state: 22, Common Neighbor accuracy: 0.1, Katz accuracy: 0.68
Random state: 23, Common Neighbor accuracy: 0.18, Katz accuracy: 0.61
Random state: 24, Common Neighbor accuracy: 0.1, Katz accuracy: 0.56
Random state: 25, Common Neighbor accuracy: 0.06, Katz accuracy: 0.72
"""

# 解析数据
lines = data.strip().split('\n')
states = [int(line.split(',')[0].split(':')[1]) for line in lines]
common_neighbor_acc = [float(line.split(',')[1].split(':')[1]) for line in lines]
katz_acc = [float(line.split(',')[2].split(':')[1]) for line in lines]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(states, common_neighbor_acc, marker='o', label='Common Neighbor accuracy')
plt.plot(states, katz_acc, marker='o', label='Katz accuracy')
plt.xlabel('Random state')
plt.ylabel('Accuracy')
plt.title('Accuracy of Common Neighbor and Katz')
plt.legend()
plt.grid(True)
plt.show()