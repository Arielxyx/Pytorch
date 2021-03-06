# coding: utf-8

import torch
import torch.nn as nn

# ----------------------------------- L1 Loss

# 生成网络输出 以及 目标输出
output = torch.ones(2, 2, requires_grad=True)*0.8 # 创建一个张量 并设置需要计算梯度 以对它的计算进行追踪
print(output)
target = torch.ones(2, 2)
print(target)

# 设置三种不同参数的L1Loss
reduce_False = nn.L1Loss(size_average=True, reduce=False)
size_average_True = nn.L1Loss(size_average=True, reduce=True)
size_average_False = nn.L1Loss(size_average=False, reduce=True)

o_0 = reduce_False(output, target)
o_1 = size_average_True(output, target)
o_2 = size_average_False(output, target)

print('\nreduce=False, 输出同维度的loss:\n{}\n'.format(o_0))
print('size_average=True，\t求平均:\t{}'.format(o_1))
print('size_average=False，\t求和:\t{}'.format(o_2))
