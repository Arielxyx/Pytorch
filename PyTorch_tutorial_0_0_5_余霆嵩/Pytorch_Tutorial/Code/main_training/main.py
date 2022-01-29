# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("..")
from Code.utils.utils import MyDataset, validate, show_confMat
from tensorboardX import SummaryWriter
from datetime import datetime

# 训练集 txt 所在路径
train_txt_path = os.path.join("..", "..", "Data", "train.txt")
# 验证集 txt 所在路径
valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")

# 分类名字
classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 训练集批次
train_bs = 16
# 验证集批次
valid_bs = 16
# 学习率
lr_init = 0.001
max_epoch = 1

# log 结果所在文件夹
result_dir = os.path.join("..", "..", "Result")

# 当前时间
now_time = datetime.now()
# 时间格式化
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

# 可视化log所在文件夹
log_dir = os.path.join(result_dir, time_str)
# 不存在 则创建
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
# 其中参数mean和std分别表示图像每个通道的 均值、方差序列
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
# 使用transforms.Normalize(mean, std)对图像按通道进行标准化，output=(input-mean)/std（即减去均值，再除以方差） 这样做可以加快模型的收敛速度。
normTransform = transforms.Normalize(normMean, normStd)
# 训练集 转换
trainTransform = transforms.Compose([
    transforms.Resize(32), # 缩放
    transforms.RandomCrop(32, padding=4), # 随机裁剪 先对图片的上下左右均填充上 4 个 pixel，值为0，即变成一个 40*40 的数据，然后再随机进行 32*32 的裁剪
    transforms.ToTensor(), # 图片转张量 并归一化 0-255 -> 0-1
    normTransform # 标准化
])
# 验证集 转换
validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder加载器
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# ------------------------------------ step 2/5 : 定义网络------------------------------------

# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (B, C 3->6, 32-5/2*2=28, 32-5/2*2=28)
        self.pool1 = nn.MaxPool2d(2, 2) # (B, C 6, 28/2=14, 28/2=14)
        self.conv2 = nn.Conv2d(6, 16, 5) # (B, C 6->16, 14-5/2*2=10, 14-5/2*2=10)
        self.pool2 = nn.MaxPool2d(2, 2) # (B, C 16, 10/2=5, 10/2=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


net = Net()     # 创建一个网络
net.initialize_weights()    # 初始化权值

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------

for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    # ------------------------------------ 观察模型在训练集上的表现 ------------------------------------
    for i, data in enumerate(train_loader):
        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        # 将图片数据转换成 Variable 类型，称为模型真正的输入
        inputs, labels = Variable(inputs), Variable(labels) # tensor不能反向传播，variable可以反向传播

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        # 维度为1方向上（分类结果） 最大值下标 最大值
        _, predicted = torch.max(outputs.data, 1)
        # 总数 += 当前批次样本量
        total += labels.size(0)
        # 正确数 += （预测值==标签值）的个数  torch.squeeze() 压缩数据维度 去掉维数为1的的维度
        correct += (predicted == labels).squeeze().sum().numpy()
        # 累计损失值
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息
        if i % 10 == 9:
            # 计算每十轮的平均损失值 并打印
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

            # 在一个图表中记录一个/多个标量的变化，常用于 Loss 和 Accuracy 曲线的记录。
            # 第一个参数：保存图的名称 | 第二个参数：Y轴数据 | 第三个参数：X轴数据。
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch) # 记录训练 loss
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)  # 记录 learning rate 每 step_size 轮学习率会更新一次
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch) # 记录 Accuracy

    # ------------------------------------ 每个epoch，记录梯度，权值------------------------------------
    # 绘制直方图和多分位数折线图，常用于监测权值、梯度的分布变化情况，便于诊断网络更新方向是否正确
    for name, layer in net.named_parameters():
        # 第一个参数：tag(string)- 该图的标签 | 第二个参数：values(torch.Tensor, numpy.array or string/blobname)- 用于绘制直方图的值 | 第三个参数：global_step(int)- 曲线图的 y 坐标
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 2 == 0:
        loss_sigma = 0.0
        # 分类数
        cls_num = len(classes_name)
        # 混淆矩阵
        conf_mat = np.zeros([cls_num, cls_num])
        # eval()时，框架会自动把 Batch Normalization 和 DropOut 固定住，不会取平均，而是用训练好的值
        net.eval()

        for i, data in enumerate(valid_loader):
            # 获取图片和标签
            images, labels = data
            images, labels = Variable(images), Variable(labels)

            # forward
            outputs = net(images)
            outputs.detach_()

            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor

            # 统计混淆矩阵 遍历所有样本
            for j in range(len(labels)):
                cate_i = labels[j].numpy() # 第 j 个样本 分类标签实际值
                pre_i = predicted[j].numpy() # 第 j 个样本 分类标签预测值
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')

# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
net_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(net.state_dict(), net_save_path)

conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)