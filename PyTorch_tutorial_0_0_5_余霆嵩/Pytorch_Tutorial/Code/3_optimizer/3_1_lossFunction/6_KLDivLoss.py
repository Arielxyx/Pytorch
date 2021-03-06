# coding: utf-8

import torch
import torch.nn as nn
import numpy as np

# -----------------------------------  KLDiv loss

loss_f = nn.KLDivLoss(size_average=False, reduce=False)
loss_f_mean = nn.KLDivLoss(size_average=True, reduce=True)

# 生成网络输出 以及 目标输出
output = torch.from_numpy(np.array([[0.1132, 0.5477, 0.3390]])).float()
output.requires_grad = True
target = torch.from_numpy(np.array([[0.8541, 0.0511, 0.0947]])).float()

loss_1 = loss_f(output, target) # tensor([[-0.2314, -0.1800, -0.2553]], grad_fn=<KlDivBackward>)
loss_mean = loss_f_mean(output, target) # tensor(-0.2222, grad_fn=<KlDivBackward>)

print('\nloss: ', loss_1)
print('\nloss_mean: ', loss_mean)


# 熟悉计算公式，手动计算样本的第一个元素的loss，注意这里只有一个样本，是 element-wise计算的

output = output[0].detach().numpy() # 第一个样本 输出 output = output[0] = [0.1132, 0.5477, 0.3390]
print('output',output)
output_1 = output[0] # 第一个样本 分类0对应的输出 0.1132 = x
print('output_1',output_1)
target_1 = target[0][0].numpy() # 第一个样本 真实分类类别 target_1 = target[0][0] = 0.8541 = y
print('target_1',target_1)

loss_1 = target_1 * (np.log(target_1) - output_1)

print('\n第一个样本第一个元素的loss：', loss_1)



