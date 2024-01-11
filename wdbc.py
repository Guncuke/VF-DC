import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# 加载WDBC数据集
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features.values
y = breast_cancer_wisconsin_diagnostic.data.targets
y = y.replace({'M': 0, 'B': 1})
y = y.values
# 数据预处理：标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch的张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train.squeeze())
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test.squeeze())

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)  # 输出层的单元数等于类别数
        # 注意，不再需要Sigmoid激活函数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 初始化模型
input_size = X_train.shape[1]
model = SimpleNN(input_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
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
    test_outputs = model(X_test)
    _, predictions = torch.max(test_outputs, 1)

# 计算准确率
accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
print(f'Test Accuracy: {accuracy:.4f}')
