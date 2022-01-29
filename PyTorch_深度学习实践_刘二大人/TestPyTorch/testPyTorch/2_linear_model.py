# @author: Ariel
# @time: 2021/3/24 10:43

# import necessary library to draw the graph - 导入绘图的包
import numpy as np
import matplotlib.pyplot as plt

# 准备数据集 - prepare the data set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 训练出模型 - define the model
def forward(x):
    # w 为 穷举的每个权重
    return x * w

# 对结果进行评估 - define the loss function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

# 穷举所有可能的权重 - [0.0, 4.0] 间隔为0.1 - [0.0, 0.1, 0.2, 0.3, ... , 4.0]
w = np.arange(0.0, 4.1, 0.1)
l_sum = 0
# 打包数据集成元组 进行遍历
for x_val, y_val in zip(x_data, y_data):
    # 用于后面打印预测值
    y_pred_val = forward(x_val)
    # 计算损失值
    loss_val = loss(x_val,y_val)
    # 损失值求和（暂时未做均值操作 - mean）
    l_sum += loss_val
    print('\nx_val', x_val)
    print('y_val', y_val)
    print('y_pred_val\n', y_pred_val)
    print('loss_val\n', loss_val)

# 一维数组的每个元素：每个权重 - 所有样本点的平均损失值MSE
plt.plot(w,l_sum/3)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()


# # @author: Ariel
# # @time: 2021/3/24 10:43
#
# # import necessary library to draw the graph
# import numpy as np
# import matplotlib.pyplot as plt
#
# # prepare the data set - 准备数据集
# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]
#
# # define the model - 训练出模型
# def forward(x):
#     # w 为 穷举的每个权重
#     return x * w
#
# # define the loss function - 对结果进行评估
# def loss(x,y):
#     y_pred = forward(x)
#     return (y_pred-y)*(y_pred-y)
#
# # 各个节点的权重
# w_list = []
# # 各个节点权重对于的损失值
# mse_list = []
#
# # 穷举所有可能的权重 - [0.0, 4.0] 间隔为0.1 - [0.0, 0.1, 0.2, 0.3, ... , 4.0]
# for w in np.arange(0.0, 4.1, 0.1):
#     print('w = ', w)
#     l_sum = 0
#
#     # 打包数据集成元组 进行遍历
#     for x_val, y_val in zip(x_data, y_data):
#         # 用于后面打印预测值
#         y_pred_val = forward(x_val)
#         # 计算损失值
#         loss_val = loss(x_val,y_val)
#         # 损失值求和（暂时未做均值操作 - mean）
#         l_sum += loss_val
#         print('\t', x_val, y_val, y_pred_val, loss_val)
#
#     print('mse =', l_sum/3)
#
#     w_list.append(w)
#     mse_list.append(l_sum/3)
#
# plt.plot(w_list,mse_list)
# plt.ylabel('loss')
# plt.xlabel('w')
# plt.show()