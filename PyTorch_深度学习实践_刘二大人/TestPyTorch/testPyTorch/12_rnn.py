# @author: Ariel
# @time: 2021/7/13 23:20

import torch

# 参数定义
seq_len = 3 # 序列数 横向
batch_size = 1 # 每个批次的样本数 纵向
input_size = 4 # 输入层维度
hidden_size = 2 # 隐藏层维度 输出层维度

num_layers = 1 ### 新增参数 RNN层数

### 定义RNN - 新增实参 RNN层数
cell = torch.nn.RNN(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size)

# 输入数据 inputs = x1 x2 x3 (batch_size个这样的样本数量)
inputs = torch.randn(seq_len, batch_size, input_size) # randn是随机生成服从正态分布的数据，返回值为张量
### h0 - 新增实参 RNN层数
hidden = torch.zeros(num_layers, batch_size, hidden_size)

### （自动）实现RNN的功能 - 输入的是整个inputs | （手动）RNNCell - 输入的是单个input 循环seq_len次 执行seq_len次RNNCell
output, hidden = cell(inputs, hidden)

print('Output size: ', output.shape) # torch.Size([3, 1, 2])
print('Output: ', output)
print('Hidden size: ', hidden.shape) # torch.Size([1, 1, 2])
print('Hidden: ', hidden)