# @author: Ariel
# @time: 2021/7/10 16:40

# 0. 导包
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 1. 准备数据集
batch_size = 64
# 使输入数值：更小 [-1,1] 正态分布
transform = transforms.Compose([
    transforms.ToTensor(), # 图像 Image -> 张量 Tensor
    transforms.Normalize((0.1307, ), (0.3081, )) # 正态分布
])

# 训练集
train_dataset = datasets.MNIST(root='../dataset/mnist', # 数据集加载路径
                               train=True, # 是否是训练集
                               download=True, # 是否允许网上下载
                               transform=transform) # 是否进行变换
# 训练集加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 测试集
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=True,
                              download=True,
                              transform=transform)
# 测试集加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 2. 构造模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784) # reshape
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)

model = Net()

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

# 训练函数
def train(epoch):
    # 损失值累计
    running_loss = 0

    # 每轮训练的内部循环函数
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data

        # 前馈
        outputs = model(inputs)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        # 反馈
        loss.backward()
        # 更新
        optimizer.step()

        running_loss += loss.item()
        # 每300批次 输出 1次平均损失值
        if batch_idx % 300 == 299:
            # 轮数 批次 平均损失值
            print("[%d, %5d] loss: %.3f" % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

# 测试函数
def test():
    # 正确数、总数 计算准确率
    correct = 0
    total = 0
    # 无需反向传播 不需要计算梯度
    with torch.no_grad():
        # 循环遍历测试集
        for data in test_loader:
            # 测试集 分为 输入图片、标签
            images, labels = data
            # 正向传播
            outputs = model(images)
            # 求得维度为1/每行的 最大值、最大值下标
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print('Accuracy on test set: %d%%' % (100*correct/total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()