import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo


breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=94)

X = breast_cancer_wisconsin_diagnostic.data.features.values
y = breast_cancer_wisconsin_diagnostic.data.targets.values

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
device = 'cuda:1'

X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train.squeeze()).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test.squeeze()).to(device)

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

input_size = X_train.shape[1]




num_epochs = 2000
accs = []

for exp in range(20):
    acc = []
    model = SimpleNN(input_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    indices_1 = torch.nonzero(y_train == 1).squeeze()
    indices_0 = torch.nonzero(y_train == 0).squeeze()
    random_indices_1 = torch.randperm(len(indices_1))
    random_indices_0 = torch.randperm(len(indices_0))
    indices_0 = indices_0[random_indices_0]
    indices_1 = indices_1[random_indices_1]
    X_train_ = X_train[indices_0[:10]]
    y_train_ = y_train[indices_1[:10]]

    i = 0
    for epoch in range(num_epochs):

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predictions = torch.max(test_outputs, 1)

        accuracy = accuracy_score(y_test.cpu().numpy(), predictions.cpu().numpy())
        acc.append(accuracy)

        model.train()
        # if (i+1)*20 > len(X_train):
        #     i=0
        outputs = model(X_train_.to(device))
        loss = criterion(outputs, y_train_.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    acc = torch.tensor(acc)
    accs.append(acc)

accs = torch.stack(accs)
mean = torch.mean(accs, dim=0)
std = torch.std(accs, dim=0)
torch.save([mean, std], 'corset_spambase.pt')