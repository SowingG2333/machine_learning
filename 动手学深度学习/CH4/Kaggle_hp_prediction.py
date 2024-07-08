import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler

# 读取训练集和测试集
train_data = pd.read_csv(r'D:\Code\机器学习\深度学习\machine_learning\动手学深度学习\kaggle\train.csv')
test_data = pd.read_csv(r'D:\Code\机器学习\深度学习\machine_learning\动手学深度学习\kaggle\test.csv')

print(train_data.shape)
print(test_data.shape)

# 删除训练集中第一列无用的ID
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数据预处理，根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 数据预处理，用独热编码处理离散文字特征“Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features)

# 将特征全部转化为数值
num_train = train_data.shape[0]
train_features = np.array(all_features[:num_train].values, dtype=np.float32)
train_labels = np.array(train_data.SalePrice.values.reshape(-1, 1), dtype=np.float32)
test_features = np.array(all_features[num_train:].values, dtype=np.float32)

# 标准化标签
scaler_labels = StandardScaler()
train_labels = scaler_labels.fit_transform(train_labels)

# 将得到的数据集从np数组转化为pytorch张量
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

# 将它们移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_features_tensor = train_features_tensor.to(device)
train_labels_tensor = train_labels_tensor.to(device)
test_features_tensor = test_features_tensor.to(device)

print(train_features_tensor.shape)

# 定义网络
num_inputs, num_outputs, num_hiddens = train_features_tensor.shape[1], 1, 128
net = nn.Sequential(
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.BatchNorm1d(num_hiddens),
    nn.Linear(num_hiddens, num_outputs)
)

# 初始化网络
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

net.apply(init_weight)
net.to(device)

# 定义损失函数
loss = nn.MSELoss()

# 定义训练超参数
batch_size, lr, num_epochs = 50, 0.002, 30
trainer = torch.optim.Adam(net.parameters(), lr=lr)

# 定义数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

data_iter = load_array((train_features_tensor, train_labels_tensor), batch_size)

loss_sum = 0

# 训练模型
for epoch in range(num_epochs):
    for X, y in data_iter:
        net.train()
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 梯度裁剪
        trainer.step()
    
    net.eval()
    with torch.no_grad():
        l = loss(net(train_features_tensor), train_labels_tensor)
        loss_sum = loss_sum + l
    print(f'epoch {epoch + 1}, loss {l:f}')

print('average loss: ')
print(loss_sum / num_epochs)

# 预测阶段
net.eval()
with torch.no_grad():
    test_preds = net(test_features_tensor).cpu().numpy()
    test_preds = scaler_labels.inverse_transform(test_preds)  # 反标准化

# 导出预测结果到CSV
submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_preds.flatten()
})

# 保存到指定位置
submission.to_csv(r'D:\Code\机器学习\深度学习\machine_learning\动手学深度学习\kaggle\submission.csv', index=False)
