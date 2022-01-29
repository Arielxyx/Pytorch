# @author: Ariel
# @time: 2021/7/15 15:10

# 引入torch
import torch
# 引入time计时
import time
# 引入math数学函数
import math
# 引入numpy
import numpy as np
# 引入plt
import matplotlib.pyplot as plt
# 从torch的工具的数据引入数据集，数据加载器
from torch.utils.data import Dataset, DataLoader
# 从torch的神经网络的数据的rnn中引入包装填充好的序列。作用是将填充的pad去掉，然后根据序列的长短进行排序
from torch.nn.utils.rnn import pack_padded_sequence
# 引入gzip 压缩文件
import gzip
# 引入csv模块
import csv

# 隐层数是100
HIDDEN_SIZE = 100
# batch的大小时256
BATCH_SIZE = 256
# 应用2层的GRU
N_LAYER = 2
# 循环100
N_EPOCHS = 100
# 字符数量时128
N_CHARS = 128
# 不使用GPU
USE_GPU = False


# 定义名字数据集的类，继承自数据集
class NameDataset(Dataset):
    # 自身初始化，是训练集为真
    def __init__(self, is_train_set=True):
        # 文件名是训练集。如果训练为真，否则是测试集
        filename = 'data/names_train.csv.gz' if is_train_set else 'data/names_test.csv.gz'
        # 用gzip打开文件名，操作text文本的时候使用'rt'，作为f
        with gzip.open(filename, 'rt') as f:
            # 阅读器是用csv的阅读器阅读文件
            reader = csv.reader(f)
            # 将文件设成一个列表
            rows = list(reader)
        # 自身名字是文件的第一列都是名字，提取第一列，对于r在rs中时
        self.names = [row[0] for row in rows]
        # 长度是名字的长度
        self.len = len(self.names)
        # 国家是第二列
        self.countries = [row[1] for row in rows]
        # print('self.countries: ', self.countries)
        # 将国家变成集合，去除重复的元素，然后进行排序，然后接着再变回列表
        self.country_list = list(sorted(set(self.countries)))
        # 得到国家的词典，将列表转化成词典(有索引)
        self.country_dict = self.getCountryDict()
        # 长度是国家的长度
        self.country_num = len(self.country_list)

    # 定义 获得项目类,提供索引访问，自身，索引
    def __getitem__(self, index):
        # 返回 带索引的名字，带索引的国家，代入，得到带国家的词典
        return self.names[index], self.country_dict[self.countries[index]]

    # 定义长度
    def __len__(self):
        # 返回长度
        return self.len

    # 定义获得国家词典
    def getCountryDict(self):
        # 现设一个空字典
        country_dict = dict()
        # idx表示进行多少次的迭代，country_name是国家名，用列举的方法将国家列表的数据提取出来，从0开始
        for idx, country_name in enumerate(self.country_list, 0):
            # 构造键值对，将国家名代入国家列表中等于1，2，3.
            country_dict[country_name] = idx
        # 返回国家列表
        return country_dict

    # 定义 索引返回国家字符串，自身索引
    def idx2country(self, index):
        # 返回 自身，将索引代入国家列表得到字符串
        return self.country_list[index]

    # 获得国家数量
    def getCountriesNum(self):
        # 返回自身国家数量
        return self.country_num


# 将训练集为真，代入名字数据集模型中得到训练集
trainset = NameDataset(is_train_set=True)
# 将训练集，batch的大小等于batch的大小，shuffle为真将数据打乱。代入到数据加载器中。得到训练加载器
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# 将训练集为假，代入名字数据集模型中得到测试集
testset = NameDataset(is_train_set=False)
# 将测试集，batch的大小等于batch的大小，shuffle为假不把数据打乱。代入到数据加载器中。得到测试加载器
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
# 训练集的获得国家数量得到国家数量
N_COUNTRY = trainset.getCountriesNum()


# 创建tensor
def create_tensor(tensor):
    # 如果使用GPU
    if USE_GPU:
        # 使用第一个GPU代入到设置，得到设置
        device = torch.device("cuda:0")
        # 让张量在设置里面跑
        tensor = tensor.to(device)
    # 返回张量
    return tensor


# 将RNN分类器分成一个类，继承自Module模块
class RNNClassifier(torch.nn.Module):
    # 定义自身初始化，输入的大小，隐层的大小，输出的大小，层数是1，bidirectional为真设成双向的。
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        # 父类初始化
        super(RNNClassifier, self).__init__()
        # 自身隐层等于隐层
        self.hidden_size = hidden_size
        # 自身层数等于层数
        self.n_layers = n_layers
        # 自身方向数量是如果bidirectional为真则是2，否则是1
        self.n_directions = 2 if bidirectional else 1
        # 将输入的大小和隐层的大小代入嵌入层得到自身嵌入层
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # 隐层的大小是输入，隐层的大小是输出，层数，双向代入GRU模型中，得到gru
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        # 因为是双向的，所以隐层×双向，输出的大小代入线性模型，得到,激活函数。
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    # 初始化h0，自身batch的大小
    def _init_hidden(self, batch_size):
        # 将层数×方向数，batch的大小，隐层的大小归零，得到h0
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        # 返回 创建张量的隐层
        return create_tensor(hidden)

    # 定义前馈计算，自身，输入，序列的长度
    def forward(self, input, seq_lengths):
        # 将输入进行转置，B*S--S*B
        input = input.t()
        # 输入的第二列是batch的大小
        batch_size = input.size(1)
        # 将batch的大小代入到自身初始隐层中，得到隐层的大小
        hidden = self._init_hidden(batch_size)
        # 将输入的大小代入到自身嵌入层得到嵌入层
        embedding = self.embedding(input)
        # 将嵌入层和序列的长度代入pack_padded_sequence中，先将嵌入层多余的零去掉，然后排序，打包出来，得到GRU的输入。
        gru_input = pack_padded_sequence(embedding, seq_lengths)
        # 将输入和隐层代入gru，得到输出和隐层
        output, hidden = self.gru(gru_input, hidden)
        # 如果是双向的
        if self.n_directions == 2:
            # 将隐层的最后一个和隐层的最后第二个拼接起来，按照维度为1的方向拼接起来。得到隐层
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        # 否则
        else:
            # 隐层就只有最后一个
            hidden_cat = hidden[-1]

        # 将隐层代入激活函数得到输出
        fc_output = self.fc(hidden_cat)
        # 返回输出
        return fc_output


# 定义名字到列表
def name2list(name):
    # 对于c在名字里，将c转变为ASC11值
    arr = [ord(c) for c in name]
    # 返回arr和长度
    return arr, len(arr)


# 定义制作张量 名字 国家
def make_tensors(names, countries):
    # 将名字代入到模型中得到ASC11值，对于名字在名字中，得到序列和长度
    sequences_and_lengths = [name2list(name) for name in names]
    # 将第一列取出来得到名字序列
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    # 将第二列转换成长tensor得到序列的长度
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    # 将国家变为长整型数据
    countries = countries.long()
    # 将名字序列的长度，序列长度的最大值的长整型归零。得到序列的张量
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    # 对于索引，序列和序列长度 在名字序列和名字长度中遍历，从零开始
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        # 将序列变成长张量，等于序列张量，idx是索引，第1，2，3.。。。，
        #:seq_len是按照从小到大排序的序列长度，这样就将序列复制到空序列中了。
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 将序列长度按照维度为0,进行排序，下降是真，得到序列长度和索引
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # 将索引赋值给序列张量
    seq_tensor = seq_tensor[perm_idx]
    # 将索引赋值给国家
    countries = countries[perm_idx]
    # 返回序列张量，序列长度，国家。创建tensor
    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths), \
           create_tensor(countries)


# 定义time_since模块
def time_since(since):
    # 现在的时间减去开始的时间的到时间差
    s = time.time() - since
    # Math.floor() 返回小于或等于一个给定数字的最大整数。计算分钟数
    m = math.floor(s / 60)
    # 减去分钟数乘以60就是剩下的秒数
    s -= m * 60
    # 返回分秒
    return '%dm %ds' % (m, s)


# 定义训练模型
def trainModel():
    # 损失设为0
    total_loss = 0
    # 对于i,名字和国家在训练加载器中遍历，从1开始
    for i, (names, countries) in enumerate(trainloader, 1):
        # 将名字和国家代入到make_tensors模型中得到输入，序列长度，目标
        inputs, seq_lengths, target = make_tensors(names, countries)
        # 将输入和序列长度代入到分类器中得到输出
        output = classifier(inputs, seq_lengths)
        # 将输出和目标代入到损失标准器中得到损失
        loss = criterion(output, target)
        # 梯度归零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新
        optimizer.step()
        # 损失标量相加得到总的损失
        total_loss += loss.item()
        # 如果i能被10整除
        if i % 10 == 0:
            # 以f开头表示在字符串内支持大括号内的python 表达式。将开始的时间代入time_since中得到分秒，循环次数，end是不换行加空格
            print('[{}]) Epoch {}'.format(time_since(start), epoch), end='')
            # f,i×输入的长度除以训练集的长度
            print('[{}/{}]'.format(i * len(inputs), len(trainset)), end='')
            # 总损失除以i×输入的长度，得到损失
            print('loss={}'.format(total_loss / (i * len(inputs))))
    # 返回总损失
    return total_loss


# 定义测试模型
def testModel():
    # 初始正确的为0
    correct = 0
    # 总长是测试集的长度
    total = len(testset)
    # 打印，，，
    print("evaluating trained model ...")
    # 不用梯度
    with torch.no_grad():
        # 对于i，名字和国家在测试加载器中遍历，从1开始
        for i, (name, countries) in enumerate(testloader, 1):
            # 将名字和国家代入到make_tensors模型中得到输入，序列长度，目标
            inputs, seq_lengths, target = make_tensors(name, countries)
            # 将输入和序列长度代入到分类器中得到输出
            output = classifier(inputs, seq_lengths)
            # 按照维度为1的方向，保持输出的维度为真，取输出的最大值的第二个结果，得到预测值
            pred = output.max(dim=1, keepdim=True)[1]
            # view_as将target的张量变成和pred同样形状的张量，eq是等于，预测和目标相等。标量求和
            correct += pred.eq(target.view_as(pred)).sum().item()
        # 100×正确除以错误,小数点后保留两位，得到百分比
        percent = '%.2f' % (100 * correct / total)
        # 测试集正确率
        print('Test set: Accuracy {}/{} {}%'.format(correct,total,percent))
    # 返回正确除以总数
    return correct / total


# 封装到if语句里面
if __name__ == '__main__':
    # 实例化分类器，字符的长度，隐层的大小，国家的数量，层数
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    # 如果使用GPU
    if USE_GPU:
        # 设置使用第一个GPU
        device = torch.device("coda:0")
        # 让分类器进到设置里面跑
        classifier.to(device)
    # 标准器是交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器是Adam。分类器的大部分参数，学习率是0.001
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    # 开始时时间的时间
    start = time.time()
    # 打印循环次数
    print("Training for %d epochs..." % N_EPOCHS)
    # 空列表
    acc_list = []
    # 对于循环在1到循环次数中。
    for epoch in range(1, N_EPOCHS + 1):
        # 训练模型
        trainModel()
        # 测试模型
        acc = testModel()
        # 将测试结果加到列表中
        acc_list.append(acc)

# 循环，起始是1，列表长度+1是终点。步长是1
epoch = np.arange(1, len(acc_list) + 1, 1)
# 将数据变成一个矩阵
acc_list = np.array(acc_list)
# 循环，列表
plt.plot(epoch, acc_list)
# x标签
plt.xlabel('Epoch')
# y标签
plt.ylabel('Accuracy')
# 绿色
plt.grid()
# 展示
plt.show()