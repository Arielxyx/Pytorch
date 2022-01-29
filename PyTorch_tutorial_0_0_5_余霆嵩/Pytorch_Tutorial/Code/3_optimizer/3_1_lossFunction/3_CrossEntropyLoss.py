# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import math

# ----------------------------------- CrossEntropy loss: base

# 最基本的损失函数定义
loss_f = nn.CrossEntropyLoss(weight=None, size_average=True, reduce=False)

# 生成网络输出 以及 目标输出
output = torch.ones(2, 3, requires_grad=True) * 0.5      # 假设一个三分类任务，batchsize=2，假设每个神经元输出都为0.5
target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)
# print(output)
# print(target)

# 执行损失函数
loss = loss_f(output, target)

print('--------------------------------------------------- CrossEntropy loss: base')
print('loss: ', loss)
print('由于reduce=False，所以可以看到每一个样本的loss，输出为[1.0986, 1.0986]')

# 熟悉计算公式，手动计算第一个样本
output = output[0].detach().numpy() # 第一个样本 输出 output = output[0] = [0.5 0.5 0.5] = x
# print('output',output)
output_1 = output[0] # 第一个样本 分类0对应的输出 0.5（无用）
# print('output_1',output_1)
target_1 = target[0].numpy() # 第一个样本 真实分类类别 target_1 = target[0] = 0 = class
# print('target_1',target_1)

# 第一项 x[class] = output[target_1] = 0.5
x_class = output[target_1]
# print('x_class',x_class)

# 第二项 log(∑exp(x[j]))
exp = math.e
sigma_exp_x = pow(exp, output[0]) + pow(exp, output[1]) + pow(exp, output[2]) # ∑exp(x[j])
log_sigma_exp_x = math.log(sigma_exp_x) # 取对数值

# 两项相加
loss_1 = -x_class + log_sigma_exp_x
print('---------------------------------------------------  手动计算')
print('第一个样本的loss：', loss_1)

# ----------------------------------- CrossEntropy loss: weight

# 权重 必须是 float类型的 tensor | tensor长度 = 类别个数
weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()
loss_f = nn.CrossEntropyLoss(weight=weight, size_average=True, reduce=False)

# 生成网络输出 以及 目标输出
output = torch.ones(2, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)

# 执行损失函数
loss = loss_f(output, target)

print('\n\n--------------------------------------------------- CrossEntropy loss: weight')
print('loss: ', loss)  #
print('原始loss值为1.0986, 第一个样本是第0类，weight=0.6,所以输出为1.0986*0.6 =', 1.0986*0.6)

# ----------------------------------- CrossEntropy loss: ignore_index

# 忽略某一类别
loss_f_1 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=1) # 忽略第一个类别
loss_f_2 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=2) # 忽略第二个类别

# 生成网络输出 以及 目标输出
output = torch.ones(3, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
target = torch.from_numpy(np.array([0, 1, 2])).type(torch.LongTensor)

# 执行损失函数
loss_1 = loss_f_1(output, target)
loss_2 = loss_f_2(output, target)

print('\n\n--------------------------------------------------- CrossEntropy loss: ignore_index')
print('ignore_index = 1: ', loss_1)     # 类别为1的样本的loss为0
print('ignore_index = 2: ', loss_2)     # 类别为2的样本的loss为0
