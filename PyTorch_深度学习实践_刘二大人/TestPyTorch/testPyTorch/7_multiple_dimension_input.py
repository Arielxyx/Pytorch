# @author: Ariel
# @time: 2021/4/6 13:30

import numpy as np
import torch
import matplotlib.pyplot as plt

# 准备数据集
# 使用numpy自动加载磁盘文件
xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)
# 取前八列   第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
x_data = torch.from_numpy(xy[:,:-1])
# 取最后一列   [-1] 最后得到的是个矩阵，如果没有中括号，拿出来的是向量
y_data = torch.from_numpy(xy[:,[-1]])

# 构造模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# 构建损失函数对象
criterion = torch.nn.BCELoss(size_average = True)
# 构建优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

epoch_lst = []
loss_lst = []

# 训练
for epoch in range(100):
    # 前馈计算
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新
    optimizer.step()

    epoch_lst.append(epoch)
    loss_lst.append(loss.item())
    print('epoch:',epoch,'loss:{:.4f}'.format(loss.item()))

# print('w =',model.linear3.weight.item())
# print('b =',model.linear3.bias.item())
plt.plot(epoch_lst,loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(ls='--')
plt.show()

