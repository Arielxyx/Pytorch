# @author: Ariel
# @time: 2021/7/14 9:41

import torch

# 准备参数
input_size = 4
hidden_size = 4
batch_size = 1
### 新增参数 RNN层数
num_layers = 1

# -------------------------------------------------------------------------------

# 准备数据集
# 构建词典
idx2char = ['e','h','l','o']
x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]

# 构建one-hot
one_hot = [[1,0,0,0],
           [0,1,0,0],
           [0,0,1,0],
           [0,0,0,1]]
one_hot_lookup = [one_hot[x] for x in x_data]

# 改变数据集及标签的形状
inputs = torch.Tensor(one_hot_lookup).view(-1,batch_size,input_size)\
### 不改变形状为 torch.LongTensor(y_data).view(-1, 1) 因为人工RNNCell需要循环labels 每次取出对应的label 所以要改成竖着的(5, 1)
labels = torch.LongTensor(y_data)

# -------------------------------------------------------------------------------

# 构建模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        ### 构建RNN
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers)

    ### 定义如何执行RNN
    def forward(self, inputs, hidden):
        # 输出out hn
        out, _ = self.rnn(inputs, hidden)
        # 输出out三维(seq_len, batch_size, hidden_size) -> 二维(seq_len x batch_size, hidden_size) 矩阵便于交叉熵
        return out.view(-1, self.hidden_size)

    ### 定义h0
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size, num_layers)

# -------------------------------------------------------------------------------

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# -------------------------------------------------------------------------------

# 训练过程
for epoch in range(100):
    # 初始化h0
    hidden = net.init_hidden()
    # 清零
    optimizer.zero_grad()
    # 前馈
    outputs = net(inputs, hidden)
    # 计算损失值：整个RNN的输出outputs(seq_len x batch_size, hidden_size)、整个标签
    loss = criterion(outputs, labels)
    # 反馈
    loss.backward()
    # 更新
    optimizer.step()

    # 挑选出输出结果hidden_size=4维度上 最大的值（分类结果） 及 最大值的下标
    _, idx = outputs.max(dim=1)
    print(idx.data)
    # 将张量Tensor转化为数组numpy
    idx = idx.data.numpy()

    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/100] loss = %.3f' % (epoch+1, loss.item()))
