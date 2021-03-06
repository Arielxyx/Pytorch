# coding: utf-8

import torch
import torch.nn as nn
import numpy as np

# ----------------------------------- Poisson NLLLoss

# 生成网络输出 以及 目标输出
log_input = torch.randn(5, 2, requires_grad=True)
print(log_input)
target = torch.randn(5, 2)
print(target)

loss_f = nn.PoissonNLLLoss()
loss = loss_f(log_input, target) # tensor(0.9580, grad_fn=<MeanBackward0>)
print('\nloss: \n', loss)

