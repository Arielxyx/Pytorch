# @author: Ariel
# @time: 2021/7/13 22:07

import torch

# 参数定义
seq_len = 3 # 序列数 横向
batch_size = 1 # 每个批次的样本数 纵向
input_size = 4 # 输入层维度
hidden_size = 2 # 隐藏层维度 输出层维度

# 定义RNNCell
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# 输入数据 dataset = inputs = x1 x2 x3 (batch_size个这样的样本数量)
dataset = torch.randn(seq_len, batch_size, input_size) # randn是随机生成服从正态分布的数据，返回值为张量
# h0
hidden = torch.zeros(batch_size, hidden_size)

# 实现RNNCell的功能
for idx, input in enumerate(dataset):
    print('='*20, idx, '='*20)
    # 某一时刻的输入xt torch.Size([1, 4])
    print('Input size: ', input.shape)

    # RNNCell
    hidden = cell(input, hidden)

    # 某一时刻的输出ht torch.Size([1, 2])
    print('Output size: ', hidden.shape)
    print(hidden)