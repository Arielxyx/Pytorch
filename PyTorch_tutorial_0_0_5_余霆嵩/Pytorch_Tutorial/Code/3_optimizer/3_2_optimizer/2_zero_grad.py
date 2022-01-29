# coding: utf-8

import torch
import torch.optim as optim

# ----------------------------------- zero_grad
# 初始化权重 w1、w2 的值 data
w1 = torch.randn(2, 2)
w1.requires_grad = True

w2 = torch.randn(2, 2)
w2.requires_grad = True

# 定义一个优化器 包含 一个参数组
optimizer = optim.SGD([w1, w2], lr=0.001, momentum=0.9)

# print(optimizer.param_groups[0])
# {'momentum': 0.9,
#  'nesterov': False,
#  'params': [tensor([[-0.3050, -0.6639],
#                     [-0.7505, -0.0964]], requires_grad=True),
#             tensor([[-0.9789,  1.4777],
#                     [ 0.8119,  1.8894]], requires_grad=True)],
#  'weight_decay': 0,
#  'dampening': 0,
#  'lr': 0.001}

# 初始化权重 w1 的梯度值 grad.data
# param_groups[0] - 第一个参数组 | ['params'] - 取 key='params' 的 value | [0] - 第一个权重 w1 | .grad - 权重是一个 Tensor（包括两个属性：权重值 data、梯度值 grad）
optimizer.param_groups[0]['params'][0].grad = torch.randn(2, 2)

print('参数w1的梯度：')
print(optimizer.param_groups[0]['params'][0].grad, '\n')  # 参数组，第一个参数(w1)的梯度

optimizer.zero_grad()
print('执行zero_grad()之后，参数w1的梯度：')
print(optimizer.param_groups[0]['params'][0].grad)  # 参数组，第一个参数(w1)的梯度
