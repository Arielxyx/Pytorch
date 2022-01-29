# @author: Ariel
# @time: 2021/7/13 23:37

import torch

# 准备参数
input_size = 4
hidden_size = 4
batch_size = 1

# -------------------------------------------------------------------------------

# 准备数据集
# 构建字典: 下标 -> 字母
idx2char = ['e','h','l','o']
# 输入字符 -> 下标（hello - 10223）
x_data = [1, 0, 2, 2, 3] # (5, 1)
# 输出字符 -> 下标（ohlol - 31232）
y_data = [3, 1, 2, 3, 2] # (5, 1)

# 简单查询: 分类值0 1 2 3 -> one-hot
one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
# 输入字符下标 -> one-hot（循环五次：hello-10223 每个字母用四维one-hot表示）
x_one_hot = [one_hot_lookup[x] for x in x_data] # (seq_len, input_size) = (5, 4)
print(x_one_hot)

# -------------------------------------------------------------------------------

# 输入字符one-hot -> 结构改为RNN训练的规定输入格式(seq_len, batch_size, input_size)
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
print(inputs)
# 输出标签直接使用分类值 -> 无需用one-hot表示 需使用LongTensor并改变结构（一个分类值对应一个one-hot）（见9_ex2）
labels = torch.LongTensor(y_data).view(-1, 1) # (5, 1)

# -------------------------------------------------------------------------------

# 构建模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        # 初始化参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # 构建RNNCell
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    # 定义如何执行一次RNNCell
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    # 定义h0
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)

net = Model(input_size, hidden_size, batch_size)

# -------------------------------------------------------------------------------

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 优化器Adam
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# -------------------------------------------------------------------------------

# 训练过程
for epoch in range(100):
    loss = 0
    # 初始化h0
    hidden = net.init_hidden()
    # 清零
    optimizer.zero_grad()
    # 前馈: 执行多次RNNCell
    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels):
        # 执行一次RNNCell
        hidden = net(input, hidden)
        # 每次计算损失值：每个RNNCell的输出hidden、单个标签（为了每次循环取出labels里对应的label label: (1, 1) -> labels: (5, 1)）
        loss += criterion(hidden, label)
        # 挑选出输出结果hidden_size=4维度上 最大的值（分类结果） 及 最大值的下标
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    # 反馈
    loss.backward()
    # 更新
    optimizer.step()
    print(', Epoch [%d/100] loss=%.4f' % (epoch+1, loss.item()))