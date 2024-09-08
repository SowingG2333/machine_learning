import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold

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

# 进行k_folds划分
dataset = TensorDataset(train_features_tensor, train_labels_tensor)

# 定义训练超参数
batch_size, lr, num_epochs, l2_lambda, k_folds = 30, 0.01, 50, 0.5, 5

kfold = KFold(n_splits=k_folds, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

net.apply(init_weight)
net.to(device)

# 定义损失函数和优化器
loss = nn.MSELoss()
trainer = torch.optim.Adam(net.parameters(), lr=lr)

# 开始训练和验证
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # 分割数据集
    train_subsampler = Subset(dataset, train_ids)
    val_subsampler = Subset(dataset, val_ids)
    
    # 创建数据加载器
    trainloader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum, n = 0.0, 0
        
        # 多次训练迭代，确保训练数据量是验证数据量的四倍
        for _ in range(4):
            for X, y in trainloader:
                X, y = X.to(device), y.to(device)
                l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                l = loss(net(X), y) + l2_lambda * l2_norm  # 计算包含正则项的损失
                trainer.zero_grad()
                l.backward()
                trainer.step()
                train_loss_sum += l.item() * y.shape[0]  # 累加损失，注意乘以y的数量以得到总损失
                n += y.shape[0]
        
        # 验证模式
        net.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for X_val, y_val in valloader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = net(X_val)
                l = loss(y_pred, y_val)
                val_loss_sum += l.item() * y_val.shape[0]
                n_val += y_val.shape[0]
            val_loss = val_loss_sum / n_val
        
        print(f'epoch {epoch + 1}, train loss {train_loss_sum / n:.6f}, val loss {val_loss:.6f}')

# 预测阶段
net.eval()
with torch.no_grad():
    test_preds = net(test_features_tensor.to(device)).cpu().numpy()

# 导出预测结果到CSV
submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_preds.flatten()
})

submission.to_csv(r'D:\Code\机器学习\深度学习\machine_learning\动手学深度学习\kaggle\submission.csv', index=False)
