import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import inv

def unravel_index(indices, shape):
    rows = indices // shape[1]
    cols = indices % shape[1]
    return rows, cols

def katz_pytorch(data, random_state):
    # 将节点写入矩阵
    matrix = np.zeros((len(data)-4, 2), dtype=int)
    for i, line in enumerate(data[4:]):
        matrix[i] = list(map(int, line.strip().split('\t')))

    # 对矩阵进行训练集和测试集的划分
    train_data, test_data = train_test_split(matrix, test_size=0.5, random_state=random_state)

    # 原矩阵的唯一节点编号
    nodes = np.unique(matrix)

    # 创建一个字典，将节点编号映射到邻接矩阵的索引
    node_to_index = {node: index for index, node in enumerate(nodes)}

    # 遍历排序后的训练矩阵，设置邻接矩阵的元素
    katz_train_adj_matrix = torch.eye(len(nodes), dtype=torch.int32).cuda()

    for row in train_data:
        i = node_to_index[row[0]]
        j = node_to_index[row[1]]
        katz_train_adj_matrix[i, j] = 1
        katz_train_adj_matrix[j, i] = 1

    # 遍历排序后的测试矩阵，设置邻接矩阵的元素
    katz_test_adj_matrix = torch.zeros((len(nodes), len(nodes)), dtype=torch.int32).cuda()

    for row in test_data:
        i = node_to_index[row[0]]
        j = node_to_index[row[1]]
        katz_test_adj_matrix[i, j] = 1
        katz_test_adj_matrix[j, i] = 1

    # 在计算最大特征值之前，将邻接矩阵转换为浮点数类型，然后计算最大特征值
    max_eigenvalue = torch.max(torch.linalg.eigvals(katz_train_adj_matrix.float()).abs())

    # 设置β值
    beta = 1 / (1 + max_eigenvalue)

    # 计算单位矩阵
    I = torch.eye(katz_train_adj_matrix.shape[0]).cuda()

    # 计算Katz指数
    katz_matrix = torch.linalg.inv(I - beta * katz_train_adj_matrix) - I

    # 使用自定义的unravel_index函数替换torch.unravel_index
    top_100_values, top_100_indices = torch.topk(katz_matrix.reshape(-1), 100, sorted=False)
    rows, cols = unravel_index(top_100_indices, katz_matrix.shape)

    # 创建一个空列表，用于存储Katz矩阵中从大到小前100个元素的索引
    new_indices = []

    # 创建一个反向映射，将索引映射回原始的节点编号
    index_to_node = {index: node for node, index in node_to_index.items()}

    # 遍历前100个索引
    for row, col in zip(*top_100_indices):
        new_indices.append((index_to_node[row.item()], index_to_node[col.item()]))

    # 与测试集中的数据进行比较，获取预测准确率
    correct_count = 0
    for row, col in new_indices:
        i = node_to_index[row]
        j = node_to_index[col]
        if katz_test_adj_matrix[i, j] == 1:
            correct_count += 1

    accuracy = correct_count / 100
    return accuracy

with open('D:\Code\机器学习\深度学习\machine_learning\节点连边预测\data\data_0.txt', 'r') as f:
    data = f.readlines()

# 定义一系列的random_state值
random_states = np.arange(0, 100)

# 初始化两个列表，用于记录每个random_state值下的预测准确率
cn_accuracies = []
katz_accuracies = []

# 对每个random_state值，运行你的代码，并记录预测准确率
for random_state in random_states:
    katz_accuracy = katz_pytorch(data, random_state)
    katz_accuracies.append(katz_accuracy)
    print(random_state)

    # 将结果写入文件
    with open('results.txt', 'a') as f:
        f.write(f'Random state: {random_state}, Katz accuracy: {katz_accuracy}\n')

# 绘制预测准确率随random_state值变化的折线图
plt.plot(random_states, cn_accuracies, label='Common Neighbor')
plt.plot(random_states, katz_accuracies, label='Katz')
plt.xlabel('Random State')
plt.ylabel('Prediction Accuracy')
plt.legend()
plt.show()