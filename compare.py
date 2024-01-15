import torch
from utils import get_dataset, get_network
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

style, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset('Spambase', 'data')
device = 'cuda:1'

vfdc_spambase = torch.load('vfdc_spambase_10ipc.pt')['data']
localdp_spambase = torch.load('localdp_spambase_10ipc.pt')['data']

vfdc_spambase_X_train = vfdc_spambase[0][0].to(device)
vfdc_spambase_y_train = vfdc_spambase[0][1].to(device)

localdp_spambase_X_train = localdp_spambase[0][0].to(device)
localdp_spambase_y_train = localdp_spambase[0][1].to(device)


vfdc_mimic = torch.load('vfdc_mimic_10ipc.pt')['data']
localdp_mimic = torch.load('localdp_mimic_10ipc.pt')['data']
# print(vfdc_mimic)

vfdc_mimic_X_train = vfdc_mimic[0][0].to(device)
vfdc_mimic_y_train = vfdc_mimic[0][1].to(device)

localdp_mimic_X_train = localdp_mimic[0][0].to(device)
localdp_mimic_y_train = localdp_mimic[0][1].to(device)

X_test = dst_test.images.to(device)
y_test = dst_test.labels.to(device)



epochs = 500
accs = []
for exp in range(20):
    net = get_network('MLP', channel, num_classes, device, im_size).to(device)  # get a random model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    acc = []
    for epoch in range(epochs):

        net.train()
        outputs = net(localdp_spambase_X_train)
        loss = criterion(outputs, localdp_spambase_y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch+1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        # 在测试集上进行预测
        net.eval()
        with torch.no_grad():
            test_outputs = net(X_test.to(device))
            _, predictions = torch.max(test_outputs, 1)

        # 计算准确率
        accuracy = accuracy_score(y_test.cpu().numpy(), predictions.cpu().numpy())
        acc.append(accuracy)
    accs.append(torch.tensor(acc))

accs = torch.stack(accs)
mean = torch.mean(accs, dim=0)
std = torch.std(accs, dim=0)
torch.save([mean, std], 'localdp_spambase.pt')
print(accs)

