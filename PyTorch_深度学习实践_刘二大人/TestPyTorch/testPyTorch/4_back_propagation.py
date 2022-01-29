# @author: Ariel
# @time: 2021/3/31 15:51

import numpy as np
import matplotlib.pyplot as plt
import torch

# 准备数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重
w = torch.Tensor([1.0])
# 显示设置权重可计算
w.requires_grad = True

# 定义模型
def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

epoch_lst = []
loss_lst = []

# 注意这里的forward(4)计算的是Tensor 需要使用item取出标量值
print('predict(before training)', 4, forward(4).item())
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        # 计算每个节点的损失值
        l = loss(x,y)
        # 反向传播 自动计算出梯度
        l.backward()
        print('\tgrad:',x,y,w.grad.item())

        # 更新权重的值
        w.data -= 0.01*w.grad.data
        # 梯度清零 防止下一次计算累加
        w.grad.data.zero_()

    print('progress:',epoch,l.item())
    epoch_lst.append(epoch)
    loss_lst.append(l.item())
print('predict(after training)', 4, forward(4).item())

plt.plot(epoch_lst,loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()