import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix

with open('data_0.txt', 'r') as f:
    data = f.readlines()

# 将节点写入矩阵
matrix = np.zeros((len(data)-1, 2), dtype=int)
for i, line in enumerate(data[4:]):
    matrix[i] = list(map(int, line.strip().split('\t')))

# 对矩阵进行训练集和测试集的划分
train_data, test_data = train_test_split(matrix, test_size=0.25, random_state=42)

# 获取训练矩阵每行第一个元素的排序索引
sort_index = np.argsort(train_data[:, 0])

# 使用排序索引对训练矩阵进行排序
sorted_train_data = train_data[sort_index]

print("排序后的训练矩阵是：\n", sorted_train_data)

# 获取训练矩阵和原矩阵共同的唯一节点编号
nodes = np.unique(np.concatenate([matrix.flatten(), sorted_train_data.flatten()]))

print("所有唯一的节点编号是：\n", nodes)

# 创建一个对角线为1的邻接矩阵
adj_matrix = np.eye(len(nodes), dtype=int)

# 创建一个字典，将节点编号映射到邻接矩阵的索引
node_to_index = {node: index for index, node in enumerate(nodes)}

# 遍历排序后的训练矩阵，设置邻接矩阵的元素
for row in sorted_train_data:
    i = node_to_index[row[0]]
    j = node_to_index[row[1]]
    adj_matrix[i, j] = 1

print("邻接矩阵是：\n", adj_matrix)

# 将邻接矩阵转换为稀疏矩阵
sparse_adj_matrix = coo_matrix(adj_matrix)

# 计算邻接矩阵的平方
adj_matrix_squared = sparse_adj_matrix.dot(sparse_adj_matrix)

# 获取邻接矩阵平方中非零元素的索引
non_zero_indices = adj_matrix_squared.nonzero()

# 获取邻接矩阵平方中非零元素的个数
non_zero_count = len(non_zero_indices[0])

print("邻接矩阵的平方是：\n", adj_matrix_squared.toarray())
print("邻接矩阵的平方中非零元素的个数是：", non_zero_count)

# 创建一个新的矩阵，矩阵中每个元素的值为其对应索引邻接矩阵平法的值除以邻接矩阵对应行和列的所有值之和减去邻接矩阵平方的值
new_matrix = np.zeros((len(nodes), len(nodes)))
for i, j in zip(*non_zero_indices):
    new_matrix[i, j] = adj_matrix_squared[i, j] / (adj_matrix[i].sum() + adj_matrix[:, j].sum() - adj_matrix[i, j])

# 创建一个全零的邻接矩阵
test_adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

# 遍历测试数据，设置邻接矩阵的元素
for row in test_data:
    i = node_to_index[row[0]]
    j = node_to_index[row[1]]
    test_adj_matrix[i, j] = 1

# 获取新矩阵中所有非零元素的索引
rows, cols = np.unravel_index(np.argsort(-new_matrix, axis=None), new_matrix.shape)

# 创建一个空列表，用于存储不在训练集中的元素的索引
new_indices = []

# 遍历所有的索引
for row, col in zip(rows, cols):
    # 检查这个元素是否在训练集中出现过
    if adj_matrix[row, col] == 0:
        # 如果没有出现过，那么将这个索引添加到新的列表中
        new_indices.append((row, col))

    # 如果新的列表中已经有100个元素，那么就可以停止遍历了
    if len(new_indices) == 100:
        break

print("新矩阵中由大到小前100个非零元素的索引是：\n", new_indices)

# 与测试集中的数据进行比较，获取预测准确率
correct_count = 0
for row, col in new_indices:
    if test_adj_matrix[row, col] == 1:
        correct_count += 1

accuracy = correct_count / 100
print("预测准确率是：", accuracy)