import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from numpy.linalg import inv
import matplotlib.pyplot as plt

def common_neighbor(data, random_state):
    # 将节点写入矩阵
    matrix = np.zeros((len(data)-4, 2), dtype=int)
    for i, line in enumerate(data[4:]):
        matrix[i] = list(map(int, line.strip().split('\t')))

    # 对矩阵进行训练集和测试集的划分
    train_data, test_data = train_test_split(matrix, test_size=0.5, random_state=random_state)

    # 原矩阵的唯一节点编号
    nodes = np.unique(matrix)

    # 创建一个对角线为1的邻接矩阵
    adj_matrix = np.eye(len(nodes), dtype=int)

    # 创建一个字典，将节点编号映射到邻接矩阵的索引
    node_to_index = {node: index for index, node in enumerate(nodes)}

    # 遍历排序后的训练矩阵，设置邻接矩阵的元素
    for row in train_data:
        i = node_to_index[row[0]]
        j = node_to_index[row[1]]
        adj_matrix[i, j] = 1

    # 将邻接矩阵转换为稀疏矩阵
    sparse_adj_matrix = coo_matrix(adj_matrix)

    # 计算邻接矩阵的平方
    adj_matrix_squared = sparse_adj_matrix.dot(sparse_adj_matrix)

    # 创建一个全零的邻接矩阵
    test_adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

    # 遍历测试数据，设置邻接矩阵的元素
    for row in test_data:
        i = node_to_index[row[0]]
        j = node_to_index[row[1]]
        test_adj_matrix[i, j] = 1

    # 将平方矩阵转换为密集矩阵（后续操作需要在密集矩阵上进行）
    adj_matrix_squared_dense = adj_matrix_squared.toarray()

    # 获取平方矩阵中由大到小前100个元素的索引
    top_100_indices = np.argpartition(-adj_matrix_squared_dense, 100, axis=None)[:100]

    # 将一维的索引数组转换为二维的索引数组
    top_100_indices = np.unravel_index(top_100_indices, adj_matrix_squared_dense.shape)

    # 创建一个空列表，用于存储平方矩阵中从大到小前100个元素的索引
    new_indices = []

    # 创建一个反向映射，将索引映射回原始的节点编号
    index_to_node = {index: node for node, index in node_to_index.items()}

    # 遍历前100个索引
    for row, col in zip(*top_100_indices):
        new_indices.append((index_to_node[row], index_to_node[col]))

    # 与测试集中的数据进行比较，获取预测准确率
    correct_count = 0
    for row, col in new_indices:
        i = node_to_index[row]
        j = node_to_index[col]
        if test_adj_matrix[i, j] == 1:
            correct_count += 1

    accuracy = correct_count / 100
    return accuracy

def katz(data, random_state):
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
    katz_train_adj_matrix = np.eye(len(nodes), dtype=int)

    for row in train_data:
        i = node_to_index[row[0]]
        j = node_to_index[row[1]]
        katz_train_adj_matrix[i, j] = 1
        katz_train_adj_matrix[j, i] = 1

    # 遍历排序后的测试矩阵，设置邻接矩阵的元素
    katz_test_adj_matrix = np.zeros((len(nodes), len(nodes),), dtype=int)

    for row in test_data:
        i = node_to_index[row[0]]
        j = node_to_index[row[1]]
        katz_test_adj_matrix[i, j] = 1
        katz_test_adj_matrix[j, i] = 1

    # 计算邻接矩阵的最大特征值
    max_eigenvalue = np.max(np.linalg.eigvals(katz_train_adj_matrix))

    # 设置β值
    beta = 1 / (1 + max_eigenvalue)

    # 计算单位矩阵
    I = np.eye(katz_train_adj_matrix.shape[0])

    # 计算Katz指数
    katz_matrix = inv(I - beta * katz_train_adj_matrix) - I

    # 获取Katz矩阵中由大到小前100个元素的索引
    top_100_indices = np.argpartition(-katz_matrix, 100, axis=None)[:100]

    # 将一维的索引数组转换为二维的索引数组
    top_100_indices = np.unravel_index(top_100_indices, katz_matrix.shape)

    # 创建一个空列表，用于存储Katz矩阵中从大到小前100个元素的索引
    new_indices = []

    # 创建一个反向映射，将索引映射回原始的节点编号
    index_to_node = {index: node for node, index in node_to_index.items()}

    # 遍历前100个索引
    for row, col in zip(*top_100_indices):
        new_indices.append((index_to_node[row], index_to_node[col]))

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
    cn_accuracy = common_neighbor(data, random_state)
    katz_accuracy = katz(data, random_state)
    cn_accuracies.append(cn_accuracy)
    katz_accuracies.append(katz_accuracy)
    print(random_state)

    # 将结果写入文件
    with open('results.txt', 'a') as f:
        f.write(f'Random state: {random_state}, Common Neighbor accuracy: {cn_accuracy}, Katz accuracy: {katz_accuracy}\n')

# 绘制预测准确率随random_state值变化的折线图
plt.plot(random_states, cn_accuracies, label='Common Neighbor')
plt.plot(random_states, katz_accuracies, label='Katz')
plt.xlabel('Random State')
plt.ylabel('Prediction Accuracy')
plt.legend()
plt.show()