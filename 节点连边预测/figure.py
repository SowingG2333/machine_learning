import matplotlib.pyplot as plt

# 数据
data = """
Random state: 40, Common Neighbor accuracy: 0.05, Katz accuracy: 0.35
Random state: 41, Common Neighbor accuracy: 0.05, Katz accuracy: 0.43
Random state: 42, Common Neighbor accuracy: 0.03, Katz accuracy: 0.41
Random state: 43, Common Neighbor accuracy: 0.12, Katz accuracy: 0.4
Random state: 44, Common Neighbor accuracy: 0.07, Katz accuracy: 0.4
Random state: 45, Common Neighbor accuracy: 0.13, Katz accuracy: 0.34
Random state: 46, Common Neighbor accuracy: 0.13, Katz accuracy: 0.2
Random state: 47, Common Neighbor accuracy: 0.07, Katz accuracy: 0.4
Random state: 48, Common Neighbor accuracy: 0.2, Katz accuracy: 0.46
Random state: 49, Common Neighbor accuracy: 0.14, Katz accuracy: 0.41

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