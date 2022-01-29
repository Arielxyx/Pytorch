# @author: Ariel
# @time: 2021/7/12 16:22

### 高级 CNN - 自定义 Inception 模块 - 仅改变通道数
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

# 准备数据集
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root="../dataset/mnist",
                                train=True,
                                transform=transform,
                                download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="../dataset/mnist",
                              train=True,
                              transform=transform,
                              download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

### Inception Module
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
        branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        outputs = [branch_pool, branch1x1, branch5x5, branch3x3]
        return torch.cat(outputs, dim=1)

# 构造训练模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)

        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

model = Net()
### 模型移至gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

epoch_lst = []
accuracy_lst = []
# 训练
def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, targets = data
        ### 用于计算的张量移至 gpu - 训练
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss+=loss
        if batch_idx%300 == 299:
            # 轮数 批次 平均损失值
            print("[%d, %5d] loss: %.3f" % (epoch + 1, batch_idx + 1, running_loss / 300))

def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            ### 用于计算的张量移至 gpu - 训练
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data,dim=1)
            correct += (predicted == labels).sum().item()
            total += outputs.size(0)
    print('Accuracy on test set: %d%%'%(100*correct/total))
    epoch_lst.append(epoch+1)
    accuracy_lst.append(100*correct/total)

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test(epoch)

    plt.plot(epoch_lst,accuracy_lst)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
