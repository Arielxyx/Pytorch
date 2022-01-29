# @author: Ariel
# @time: 2021/7/21 14:24

import torch.nn as nn
from .BasicModule import BasicModule
from torch.functional import F

# 残差块 - 避免梯度消失
class ResidualBlock(nn.Module):
    # 初始化
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock,self).__init__()
        # 仅用于模型网络训练
        self.left = nn.Sequential(
            # 卷积层
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            # 使数据分布一致
            nn.BatchNorm2d(outchannel),
            # 激励层
            nn.ReLU(inplace=True),
            # 卷积层
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            # 使数据分布一致
            nn.BatchNorm2d(outchannel)
        )
        # 抄近道
        self.right = shortcut

    def forward(self, x):
        # 正常模型训练 y
        y = self.left(x)
        # 抄近道 x
        x = x if self.right is None else self.right(x)
        # 激活前 先相加 x+y
        return nn.ReLU(x+y)

class ResNet34(BasicModule):
    def __init__(self,num_classes=2):
        super(ResNet34,self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = self._make_layer(64,128,3)
        self.layer2 = self._make_layer(128,256,4,stride=2)
        self.layer3 = self._make_layer(256,512,6,stride=2)
        self.layer4 = self._make_layer(512,512,3,stride=2)
        self.fc = nn.Linear(512,num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        short_cut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers=[]
        layers.append(ResidualBlock(inchannel,outchannel,stride,short_cut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel,stride))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)
        return self.fc(x)