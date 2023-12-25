from utils import get_network
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
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

net = get_network('MLP', 1, 2, device, im_size=(1, 714))

data_path = os.path.join('data', 'mimic')
x_train = torch.tensor(np.load(os.path.join(data_path, "x_train.npy"))).to(device)
x_test = torch.tensor(np.load(os.path.join(data_path, "x_test.npy"))).to(device)
y_train = torch.tensor(np.load(os.path.join(data_path, "y_train.npy"))).to(device)
y_test = torch.tensor(np.load(os.path.join(data_path, "y_test.npy"))).to(device)
dst_train = TensorDataset(x_train, y_train)
dst_test = TensorDataset(x_test, y_test)

trainloader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)

# Assuming you have a classification task, you can use CrossEntropyLoss as the loss function
criterion = torch.nn.CrossEntropyLoss()

# Assuming you have already defined your optimizer, for example, SGD
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 2000  # You can adjust this based on your needs

net.to(device)  # Move the network to the GPU

output = net(x_test)
print(output.shape)
acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), y_test.cpu().data.numpy()))
print(acc)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.long().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

output = net(x_test)
print(output.shape)
acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), y_test.cpu().data.numpy()))
print(acc / len(x_test))


