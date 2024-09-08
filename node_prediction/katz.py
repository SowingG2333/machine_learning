import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from numpy.linalg import inv

with open('.data/data_2.txt', 'r') as f:
    data = f.readlines()

# 将节点写入矩阵
matrix = np.zeros((len(data)-4, 2), dtype=int)
for i, line in enumerate(data[4:]):
    matrix[i] = list(map(int, line.strip().split('\t')))

# 对矩阵进行训练集和测试集的划分
train_data, test_data = train_test_split(matrix, test_size=0.25, random_state=44)

# 原矩阵的唯一节点编号
nodes = np.unique(matrix)

# 创建一个字典，将节点编号映射到邻接矩阵的索引
node_to_index = {node: index for index, node in enumerate(nodes)}

# 遍历排序后的训练矩阵，设置邻接矩阵的元素
katz_train_adj_matrix = np.zeros((len(nodes),len(nodes)), dtype=int)

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
print("β值是：", beta)

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

print("Katz矩阵中由大到小前100个非零元素的索引是：\n", new_indices)

# 与测试集中的数据进行比较，获取预测准确率
correct_count = 0
for row, col in new_indices:
    i = node_to_index[row]
    j = node_to_index[col]
    if katz_test_adj_matrix[i, j] == 1:
        correct_count += 1

accuracy = correct_count / 100
print("预测准确率是：", accuracy)