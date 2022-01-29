# @author: Ariel
# @time: 2021/4/3 16:04

import torch
import matplotlib.pyplot as plt

# 准备数据集 3*1的矩阵
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

# 构建模型 该模型类应该继承自nn（neural network神经网络）模块，是所有神经网络模块的基类
class LinearModel(torch.nn.Module):
    # 初始化模型对象时调用
    def __init__(self):
        # 调用父类构造器，传入当前模型名称、模型对象本身self
        super(LinearModel,self).__init__()
        # 构建模型对象本身的成员对象linear（可被调用），包含权重w和偏置b两个Tensor
        self.linear = torch.nn.Linear(1,1)

    # 前馈计算时调用
    def forward(self, x):
        # 计算y_hat=wx+b
        y_pred = self.linear(x) # 调用对象linear
        return y_pred

# 实例化模型
model = LinearModel()

# 构建损失对象 参数size_average = False https://blog.csdn.net/u013841196/article/details/102831670
criterion = torch.nn.MSELoss(reduction='sum')
# 构建优化器 不会构建计算图
optimizer = torch.optim.Rprop(model.parameters(),lr=0.01)

epoch_lst = []
loss_lst = []

# 训练
for epoch in range(1000):
    # 前馈计算
    y_pred = model(x_data) # 调用对象model里面的forward()函数
    loss = criterion(y_pred, y_data) # 计算损失值
    print('epoch:',epoch,'loss:',loss.item())
    # 梯度清零
    optimizer.zero_grad()
    # 反馈求梯度
    loss.backward()
    # 更新权重
    optimizer.step()

    epoch_lst.append(epoch)
    loss_lst.append(loss.item())

print('w =',model.linear.weight.item())
print('b =',model.linear.bias.item())

# 测试集
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred =',y_test.data)

plt.plot(epoch_lst,loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()