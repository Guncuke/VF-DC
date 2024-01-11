import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import os

# 假设你有一个CSV文件，包含MIMIC-III数据，其中包含特征列和标签列
# 这里只是一个示例，实际情况需要根据你的数据进行调整
data_path = os.path.join('data', 'mimic')
# x_train = F.normalize(torch.tensor(np.load(os.path.join(data_path, "x_train.npy"))), p=2, dim=0)
# x_test = F.normalize(torch.tensor(np.load(os.path.join(data_path, "x_test.npy"))), p=2, dim=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(np.load(os.path.join(data_path, "x_train.npy")))
X_test = scaler.fit_transform(np.load(os.path.join(data_path, "x_test.npy")))
y_train = torch.LongTensor(np.load(os.path.join(data_path, "y_train.npy")))
y_test = torch.LongTensor(np.load(os.path.join(data_path, "y_test.npy")))
print(X_train.shape)
print(y_train.shape)


# 转换为PyTorch的张量
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)  # 输出层的单元数等于类别数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

device = 'cuda'
# 初始化模型
input_size = X_train.shape[1]
model = SimpleNN(input_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size=1024
# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 随机打乱数据集
    permutation = torch.randperm(X_train.size()[0])

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 在测试集上进行预测
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.to(device))
    _, predictions = torch.max(test_outputs, 1)

# 计算准确率
accuracy = accuracy_score(y_test.cpu().numpy(), predictions.cpu().numpy())
print(f'Test Accuracy: {accuracy:.4f}')
