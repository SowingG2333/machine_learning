# 导入所需库
import torch
import pandas as pd
import os

# 创建一个名为 house_tiny.csv 的 CSV 文件，并在其中写入数据。
os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转换为张量格式
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, Y)

# 作业1：删除缺失值最多的列
rows = data.iloc[:, :]
missing_value_counts = rows.isnull().sum(axis=0)# 此处axis=0代表对所有的行操作求和
row_with_most_missing_values = missing_value_counts.idxmax()
rows = rows.drop(row_with_most_missing_values, axis=1)# 此处axis=1代表删除对应索引的列
print(rows)

# 作业2:将预处理后的数据集转化为张量格式
rows = rows.fillna(rows.mean())
rows_tensor = torch.tensor(rows.to_numpy(dtype=float))
print(rows_tensor)

# 如果采用删除法处理缺失值
rows_delete = data.iloc[:, :].dropna()
print(rows_delete)
rows_delete_tensor = torch.tensor(rows_delete.to_numpy(dtype=float))
print(rows_delete_tensor)