import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

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

# 将得到的数据集从np数组转化为pytorch张量
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

# 定义损失函数
loss = nn.MSELoss()

# 定义网络
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(nn.Flatten(), 
                        nn.Linear(in_features, 1024),
                       nn.ReLU(),
                       nn.Linear(1024, 256),
                       nn.ReLU(),
                       nn.Linear(256, 1))
    return net

# 定义对数损失
def log_rmse(net, features, labels):
    # 将小于一的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# 定义训练函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# 定义k折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = torch.tensor(X_part, dtype=torch.float32), torch.tensor(y_part, dtype=torch.float32)
        else:
            X_part, y_part = torch.tensor(X_part, dtype=torch.float32), torch.tensor(y_part, dtype=torch.float32)
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 验证log rmse{float(valid_ls[-1]):f}')
    # 预测阶段
    net.eval()
    with torch.no_grad():
        test_preds = net(test_features_tensor).cpu().numpy()
    # 导出预测结果到CSV
    submission = pd.DataFrame({
        "Id": test_data["Id"],
        "SalePrice": test_preds.flatten()
    })
    submission.to_csv(r'D:\Code\机器学习\深度学习\machine_learning\动手学深度学习\kaggle\submission.csv', index=False)
    return train_l_sum / k, valid_l_sum / k

# 开始训练
k, num_epochs, lr, weight_decay, batch_size = 5, 200, 0.01, 25, 128
train_l, valid_l = k_fold(k, train_features_tensor, train_labels_tensor, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')
d2l.plt.show()