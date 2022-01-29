# @author: Ariel
# @time: 2021/7/12 8:36

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

# 构造训练模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # b n w h = b 1 28 28 -> b 10 24 24
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5) # n m k1 k2
        # b n w h = b 10 24 24 -> b 10 12 12
        self.pooling = torch.nn.MaxPool2d(2) # stride = 2
        # b n w h = b 10 12 12 -> b 20 8 8
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        # b n w h = b 20 8 8 -> b 20 4 4
        self.pooling = torch.nn.MaxPool2d(2)  # stride = 2

        # 全连接层相较于 lecture_09 简化了
        self.fc = torch.nn.Linear(320, 10)

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        # print('x.shape:',x.shape)
        # 全连接层输入扁平化 x = x.view(-1, 784)
        x = x.view(batch_size, -1)
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
            # 未设置 keepdim=True 不保持输出维度(batch_size, 1) 最终维度为(batch_size)
            _, predicted = torch.max(outputs.data,dim=1)
            # print('predicted.shape: ', predicted.shape)
            # print('labels.shape: ', labels.shape)
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
