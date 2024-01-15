import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import os


data_path = os.path.join('data', 'mimic')

scaler = StandardScaler()
device = 'cuda'
X_train = scaler.fit_transform(np.load(os.path.join(data_path, "x_train.npy")))
X_test = scaler.fit_transform(np.load(os.path.join(data_path, "x_test.npy")))
y_train = torch.LongTensor(np.load(os.path.join(data_path, "y_train.npy"))).to(device)
y_test = torch.LongTensor(np.load(os.path.join(data_path, "y_test.npy"))).to(device)
print(X_train.shape)
print(y_train.shape)

X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)


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


# 初始化模型
input_size = X_train.shape[1]
model = SimpleNN(input_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

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
