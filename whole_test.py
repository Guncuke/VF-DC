from utils import get_network
from torch.utils.data import Dataset
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


gpu_num = 3
torch.cuda.set_device(gpu_num)
device = torch.device('cuda:{}'.format(gpu_num))

net = get_network('MLP', 1, 2, device, im_size=(1, 30))

# data_path = os.path.join('data', 'mimic')
# # x_train = torch.tensor(np.load(os.path.join(data_path, "x_train.npy"))).to(device)
#
# # 读取 CSV 文件
# csv_filename = os.path.join('result', 'vis_DM_MIMIC_MLP_50ipc_exp0_iter10000.csv')
# df = pd.read_csv(csv_filename)
#
# # 提取特征和标签列
# features = df.iloc[:, :-1].values  # 所有列除了最后一列是特征
# labels = df.iloc[:, -1].values  # 最后一列是标签
#
# # 将 NumPy 数组转换为 PyTorch Tensor
# x_train = torch.tensor(features, dtype=torch.float32).to(device)
# y_train = torch.tensor(labels, dtype=torch.float32).to(device)
# x_test = torch.tensor(np.load(os.path.join(data_path, "x_test.npy"))).to(device)
# # y_train = torch.tensor(np.load(os.path.join(data_path, "y_train.npy"))).to(device)
# y_test = torch.tensor(np.load(os.path.join(data_path, "y_test.npy"))).to(device)
# dst_train = TensorDataset(x_train, y_train)
# dst_test = TensorDataset(x_test, y_test)

# data_path = os.path.join('data', 'customer.csv')
# data = pd.read_csv(data_path)
# columes_to_drop = ['ID_code', 'target']
#
# x = data.drop(columns=columes_to_drop, axis=1)
# y = data.iloc[:,1]
# x_train = x.iloc[:len(data) - 10000]
# x_test = x.iloc[-10000:]
# y_train = torch.tensor(y.iloc[:len(data) - 10000].values)
# y_test = torch.tensor(y.iloc[-10000:].values)
# x_train = F.normalize(torch.tensor(x_train.values), p=2, dim=0)
# x_test = F.normalize(torch.tensor(x_test.values, dtype=torch.float), p=2, dim=0)
# # print(y_train.shape)
# dst_train = TensorDataset(x_train, y_train)
# dst_test = TensorDataset(x_test, y_test)
# fetch dataset
# fetch dataset
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
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train.squeeze()).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test.squeeze()).to(device)

dst_train = TensorDataset(X_train, y_train)
dst_test = TensorDataset(X_test, y_test)


# Assuming you have a classification task, you can use CrossEntropyLoss as the loss function
criterion = torch.nn.CrossEntropyLoss()

# Assuming you have already defined your optimizer, for example, SGD
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 1000  # You can adjust this based on your needs

  # Move the network to the GPU

for epoch in range(num_epochs):

    output = net(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print statistics
    if epoch % 10 == 9:  # Print every 10 mini-batches
        print(f'epoch:{epoch}:{loss.item()}')

output = net(X_test.to(device))
print(output.shape)
acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), y_test.cpu().data.numpy()))
print(acc / len(X_test))


