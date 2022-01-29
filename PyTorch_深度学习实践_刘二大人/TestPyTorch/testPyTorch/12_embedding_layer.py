# @author: Ariel
# @time: 2021/7/14 11:19

import torch

# 准备参数
input_size = 4
hidden_size = 8
batch_size = 1
seq_len = 5
num_layers = 2

### 稠密层
embedding_size = 10
### 分类数目
num_class = 4

# 准备数据集
# 构建词典
idx2char = ['e','h','l','o']
### 设置了batch_first 所以要交换位置
x_data = [[1,0,2,2,3]] # (seq_len,batch_size) -> (batch_size,seq_len)
y_data = [3,1,2,3,2]

### Embedding层 输入使用LongTensor (batch_size,seq_len)实质包含了input_size 且有利于后续做交叉熵求损失值（二维矩阵）
inputs = torch.LongTensor(x_data) # (batch_size,seq_len,input_size)
labels = torch.LongTensor(y_data)

# 构建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        ### 嵌入层
        self.emb = torch.nn.Embedding(input_size,
                                      embedding_size)
        ### RNN
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        ### 线性层
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, batch_size, hidden_size)
        # (batch_size,seq_len,input_size) -> (batch_size,seq_len,embedding_size)
        x = self.emb(x)
        # (batch_size,seq_len,embedding_size) -> (batch_size,seq_len,hidden_size)
        x, _ = self.rnn(x, hidden)
        # (batch_size,seq_len,hidden_size) -> (batch_size,seq_len,num_class)
        x = self.fc(x)
        # (batch_size,seq_len,num_class) -> (batch_size * seq_len,num_class)
        return x.view(-1,num_class)

net = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    # torch.Size([5, 4])
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 选取这五行每行 最大值_、最大值下标
    _, idx = outputs.max(dim = 1)
    # print('idx: ', idx)
    idx = idx.data.numpy()
    # print('idx numpy: ', idx)


    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/100] loss = %.3f' % (epoch+1, loss.item()))