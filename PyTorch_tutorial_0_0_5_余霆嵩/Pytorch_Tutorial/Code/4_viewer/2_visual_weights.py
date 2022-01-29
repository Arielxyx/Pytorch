# coding: utf-8
import os
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
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
pretrained_dict = torch.load(os.path.join("..", "2_model", "net_params.pkl"))
net.load_state_dict(pretrained_dict)

writer = SummaryWriter(log_dir=os.path.join("..", "..", "Result", "visual_weights"))
params = net.state_dict()
for k, v in params.items():
    if 'conv' in k and 'weight' in k:
        # conv1.weight torch.Size([6, 3, 5, 5])
        # conv2.weight torch.Size([16, 6, 5, 5])
        c_int = v.size()[1]     # 输入层通道数 = 3 | 6
        c_out = v.size()[0]     # 输出层通道数 = feature map = 6 | 16

        # ------------- 某一类卷积核 权重可视化 -----------------------------------------
        # 遍历所有种类数的卷积核 以out/feature map为基准单位，绘制一组卷积核，一张feature map对应的卷积核个数为输入通道数
        for j in range(c_out):
            print(k, v.size(), j)
            # 每次以输入通道数为一组 压缩维度，只取第0个维度的第j个、插入第1个维度 [6, 3, 5, 5]->[3, 5, 5]->[3, 1, 5, 5]，为make_grid制作输入
            kernel_j = v[j, :, :, :].unsqueeze(1)
            # make_grid tensor->[B=输入通道数=3, C=1, H, W] | nrow->输入通道数
            kernel_grid = vutils.make_grid(kernel_j, normalize=True, scale_each=True, nrow=c_int) # 1*输入通道数, w, h
            # j 表示feature map数
            writer.add_image(k+'_split_in_channel', kernel_grid, global_step=j)

        # ------------- 所有种类卷积核 权重可视化 -----------------------------------------
        # 将一个卷积层的卷积核绘制在一起，每一行是一个feature map的卷积核
        k_w, k_h = v.size()[-1], v.size()[-2]
        # [6, 3, 5, 5]->[6x3, 1, 5, 5]
        kernel_all = v.view(-1, 1, k_w, k_h)
        # make_grid tensor->[B=输出通道数x输入通道数, C, H, W] | nrow->输入通道数
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=c_int)  # 1*输入通道数, w, h
        writer.add_image(k + '_all', kernel_grid, global_step=666)
writer.close()