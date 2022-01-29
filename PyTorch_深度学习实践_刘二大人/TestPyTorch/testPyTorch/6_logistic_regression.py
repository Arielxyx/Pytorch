# @author: Ariel
# @time: 2021/4/5 12:00

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 准备数据集
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

# 构造模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()

# 创建损失对象
criterion = torch.nn.BCELoss(size_average=False)
# 创建优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

epoch_lst = []
loss_lst = []

# 训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)

    print('epoch:',epoch,'loss: {:.4f}'.format(loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_lst.append(epoch)
    loss_lst.append(loss.item())

print('w =',model.linear.weight.item())
print('b =',model.linear.bias.item())

# ------------------------------------------------------
# 测试-1
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred =',y_test.data)

plt.plot(epoch_lst,loss_lst)
plt.title('BCE')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(ls='--') # 生成网格
plt.show()

# ------------------------------------------------------
# 测试-2
# 创建等差数列 0-10h 采集200个点
x = np.linspace(0,10,200)
print('x',x)
# 类似于numpy里的reshape 用于改变数组格式 200行*1列
x_t = torch.Tensor(x).view((200,1))
print('x_t',x_t)

# 使用模型预测y_t的值
y_t = model(x_t)
print('y_t',y_t)
# 将Tensor矩阵转化为数组
y = y_t.data.numpy()
print('y',y)

plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('hours')
plt.ylabel('probability of pass')
plt.grid()
plt.show()