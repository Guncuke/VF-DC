import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from utils import get_dataset
import random

device = 'cuda:1'
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=94)

X = breast_cancer_wisconsin_diagnostic.data.features.values
y = breast_cancer_wisconsin_diagnostic.data.targets.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        d = 256
        delta = torch.tensor(0.001, dtype=torch.float)
        epsilon = torch.tensor(25, dtype=torch.float)
        sigma = torch.sqrt(2 * d * torch.log(1.25 / delta)) / epsilon
        noise = torch.randn(256) * sigma
        self.noise = noise.to('cuda:1')


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

input_size = 57

# style, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset('Spambase', 'data')
#
# X_test = dst_test.images.to(device)
# y_test = dst_test.labels.to(device)
# ''' organize the real dataset '''
# images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
# labels_all = [dst_train[i][1] for i in range(len(dst_train))]
#
#
# images_all = torch.cat(images_all, dim=0).to(device)
# labels_all = torch.tensor(labels_all, dtype=torch.long).to(device)
#
# indices_class = [[] for c in range(num_classes)]
# for i, lab in enumerate(labels_all):
#     indices_class[lab].append(i)
# for c in range(num_classes):
#     print('class c = %d: %d real images' % (c, len(indices_class[c])))
#
# indices = []
# label_syn = []
# for label, indice in enumerate(indices_class):
#     label_syn += [label] * 10
#     indice = random.sample(indice, 10)
#     indices += indice
#
# y_train = torch.LongTensor(label_syn).to(device)
# X_train = images_all[indices].to(device)

num_epochs = 500
accs = []

for exp in range(20):
    acc = []
    model = SimpleNN(input_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00003)

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
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))

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
torch.save([mean, std], 'whole.pt')