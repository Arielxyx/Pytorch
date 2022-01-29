# @author: Ariel
# @time: 2021/3/24 17:39

import numpy as np
import matplotlib.pyplot as plt

# 准备数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重
w = 1.0

# 定义模型 y_hat = x*w
def forward(x):
    return x*w

# 整体的损失函数
def cost(x_lst, y_lst):
    cost_val = 0
    for x, y in zip(x_lst,y_lst):
        y_pred = forward(x)
        cost_val+=(y_pred-y)**2
    return cost_val/len(x_lst)

# 整体的梯度函数
def gradient(x_lst,y_lst):
    grad_val = 0
    for x,y in zip(x_lst,y_lst):
        grad_val += 2*x*(forward(x)-y)
    return grad_val/len(x_lst)

# 保存epoch
epoch_list = []
# 保存每轮epoch对应的损失值cost
cost_list = []

print('Predict（before training）', 4, forward(4))
# 训练100轮
for epoch in range(100):
    cost_val = cost(x_data,y_data)
    grad_val = gradient(x_data,y_data)
    # 更新权重
    w -= 0.01*grad_val
    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print('epoch :', epoch, '| w =', w, '| cost =', cost_val)
print('Predict（after training）', 4, forward(4))

plt.plot(epoch_list,cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.grid(ls='--')
plt.show()