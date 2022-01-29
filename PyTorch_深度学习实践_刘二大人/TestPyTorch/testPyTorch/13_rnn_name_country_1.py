# @author: Ariel
# @time: 2021/7/15 15:34

import torch
import numbers as np
import matplotlib.pyplot as plt
# 数据集、加载数据集
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# gzip 解压缩
import gzip
# csv 文件内容读取器
import csv

# 从torch的神经网络的数据的rnn中引入包装填充好的序列。作用是将填充的pad去掉，然后根据序列的长短进行排序
from torch.nn.utils.rnn import pack_padded_sequence

# time计时
import time
# math数学函数
import math

# 准备参数
# 每个批次的 大小/样本量
BATCH_SIZE = 256
HIDDEN_SIZE = 100
EMBEDDING_SIZE = HIDDEN_SIZE

N_LAYER = 2
USE_GPU = True

N_CHARS = 128
INPUT_SIZE = N_CHARS

N_EPOCHS = 100

# 准备数据集
class NameDataset(Dataset):
    def __init__(self, is_train_set = True):
        # 获取训练集/测试集 的 文件名
        filename = 'data/names_train.csv.gz' if is_train_set else 'data/names_test.csv.gz'
        # gzip打开压缩包 操作text文本的时候使用'rt'，作为f
        with gzip.open(filename, 'rt') as f:
            # csv读取文件内容
            reader = csv.reader(f)
            # 读取的所有行（名字 - 输入、国家 - 标签） 存至 列表
            rows = list(reader)

        # 名字 - 输入列表
        self.names = [row[0] for row in rows]
        # 国家 - 标签列表
        self.countries = [row[1] for row in rows]

        # 样本数据量（多少条样本数据）
        self.len = len(self.names)

        # 使用set获取不重复的国家 排序后 存至列表（key）
        self.country_list = list(sorted(set(self.countries)))
        # 构建词典 {key: value}
        self.country_dict = self.getCountryDict()
        # 词典长度
        self.country_num = len(self.country_list)

    # 构建词典：根据国家 获取 索引
    def getCountryDict(self):
        # 空字典
        country_dict = dict()
        # 循环不重复 and 排序的国家列表
        for idx, country_name in enumerate(self.country_list,0):
            # {国家名字country_name：索引idx}
            country_dict[country_name] = idx
        return country_dict

    # 根据索引 获取 国家
    def idx2country(self, idx):
        return self.country_list[idx]

    def __getitem__(self, item):
        # 样本数据中 对应样本下标的 名字、国家的词典索引
        return self.names[item], self.country_dict[self.countries[item]]

    def __len__(self):
        # 样本数据量（多少条样本数据）
        return self.len

    def getCountriesNum(self):
        # 词典长度
        return self.country_num

trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, shuffle=True, batch_size=BATCH_SIZE)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, shuffle=False, batch_size=BATCH_SIZE)

# 词典长度 - 分类数
N_COUNTRY = trainset.getCountriesNum()
OUTPUT_SIZE = N_COUNTRY

# ------------------------------------------------------------------------------

# 判定是否使用gpu
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

# 设计模型
class RNNClassifier(torch.nn.Module):
    # 初始化参数、模型
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers = 1, bidirectional = True):
        super(RNNClassifier,self).__init__()
        # 嵌入层
        self.embedding_size = embedding_size
        # 隐藏层
        self.hidden_size = hidden_size
        # GRU层数
        self.n_layers = n_layers
        # 双向 or 单向
        self.n_directions = 2 if bidirectional else 1

        # 嵌入层 input_size, embedding_size=hidden_size
        self.embedding = torch.nn.Embedding(input_size,
                                            embedding_size)
        # GRU
        self.gru = torch.nn.GRU(num_layers=n_layers,
                                input_size=embedding_size,
                                hidden_size=hidden_size,
                                bidirectional=bidirectional)
        # Linear
        self.fc = torch.nn.Linear(hidden_size * self.n_directions,
                                  output_size)

    # 初始化 h0
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,
                           batch_size,
                           self.hidden_size)
        return create_tensor(hidden)

    # 前馈计算
    def forward(self, inputs, seq_lengths):
        # 矩阵转置 (batch_size, seq_len, [input_size]) -> (seq_len, batch_size, [input_size])
        inputs = inputs.t()

        # 初始化 h0
        batch_size = inputs.size(1)
        hidden = self.init_hidden(batch_size)

        # 嵌入层 (seq_len, batch_size) -> (seq_len, batch_size, embedding_size)
        embedding = self.embedding(inputs)

        # GRU
        # GRU格式的输入：将嵌入层、序列长度代入pack_padded_sequence中，先将嵌入层多余的零去掉，打包出来，得到GRU的输入
        gru_input = pack_padded_sequence(embedding, seq_lengths)
        # 执行GRU
        # hidden: (n_layers*n_directions, batch_size, hidden_size)
        output, hidden = self.gru(gru_input, hidden)

        # 如果为双向神经网络 要进行拼接
        if self.n_directions == 2:
            # 将隐层的最后一个和隐层的最后第二个拼接起来，按照维度为1的方向拼接起来 得到隐层
            # (n_layers*n_directions, batch_size, hidden_size) -> (batch_size, hidden_size*n_directions)
            hidden_cat = torch.cat([hidden[-1],hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        # Linear (batch_size, hidden_size*n_directions) -> (batch_size, output_size)
        fc_output = self.fc(hidden_cat)
        return fc_output

# ------------------------------------------------------------------------------

def name2list(name):
    # 将一个名字的 每个字符 转化为 ASCII
    arr = [ord(c) for c in name]
    # 返回名字的 (ASCII序列、序列长度)
    return arr, len(arr)

def make_tensors(names, countries):
    # 所有样本的 names名字列表 转化为 (ASCII序列、序列长度)列表
    sequences_and_length = [name2list(name) for name in names]
    # ASCII序列 列表
    name_sequences = [sl[0] for sl in sequences_and_length]
    # 序列长度 列表
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_length]) # seq_lengths和seq_tensor做运算 所以都是LongTensor
    # 将国家 变为 长整型数据
    # 加载数据集时 根据元素下标 顺序获取列表的每个元素：隐式调用__getitem__方法 国家字符串 -> 国家分类索引
    countries = torch.LongTensor(countries.long()) # 因为国家为结果分类标签 所以应为LongTensor 将每个数据改为long

    # 填充 padding
    # 先制作一个全零张量 (名字ASCII序列的个数/样本的个数, 最大的序列长度)
    seq_tensor = torch.zeros(len(name_sequences),seq_lengths.max()).long() # 输入seq_tensor为LongTensor 所以数据均为long
    # 将实际数据复制到全零张量中
    for idx, (name_seq, seq_len) in enumerate(zip(name_sequences,seq_lengths),0):
        # 第idx个名字样本 序列[0,seq_len) 赋值为 实际数据
        seq_tensor[idx, :seq_len] = torch.LongTensor(name_seq) # 模型包含嵌入层 所以输入seq_tensor为LongTensor 隐含input_size(one-hot)

    # 排序
    # 降序排列后的 序列长度列表, 原序列下标
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # 按照原序列下标 查询 原序列张量 -> 降序时的序列张量
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)

# ------------------------------------------------------------------------------

def trainModel():
    total_loss = 0
    # 加载数据集时 根据元素下标 顺序获取列表的每个元素：隐式调用__getitem__方法 国家字符串 -> 国家分类索引
    for i, (names, countries) in enumerate(trainloader, 1):
        # 数据处理 将字符串列表 -> 填充排序后的 ASCII序列张量(batch_size, seq_len, [input_size])、序列长度张量(batch_size, 1)、分类张量(batch_size, 1)
        inputs, seq_lengths, target = make_tensors(names, countries)
        # 前馈训练 ASCII序列张量(batch_size, seq_len, [input_size]) -> (batch_size, output_size)
        output = classifier(inputs, seq_lengths)
        # 求损失值 (batch_size, output_size)、分类张量(batch_size, 1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

        if i%10 == 0:
            # 将开始的时间代入time_since中得到分秒，循环次数，end是不换行加空格
            print('[{}]) Epoch {}'.format(time_since(start), epoch), end='')
            # （已训练数据量 = 批次i*每批次输入数据量） / 训练集的长度
            print('[{}/{}]'.format(i * len(inputs), len(trainset)), end='')
            # 平均损失 = 总损失 ÷ （已训练数据量 = 批次i*每批次输入数据量）
            print('loss={}'.format(total_loss / (i * len(inputs))))

def testModel():
    # 初始正确的为0
    correct = 0
    # 总长是测试集的长度
    total = len(testset)
    print("evaluating trained model ...")

    # 不用梯度
    with torch.no_grad():
        for i, (name, countries) in enumerate(testloader, 1):
            # 数据处理 将字符串列表 -> 填充排序后的 ASCII序列张量(batch_size, seq_len, [input_size])、序列长度张量(batch_size, 1)、分类张量(batch_size, 1)
            inputs, seq_lengths, target = make_tensors(name, countries)
            # (batch_size, seq_len, [input_size]) -> (batch_size, output_size) torch.Size([256, 18])
            output = classifier(inputs, seq_lengths)

            # 按照维度为1的方向，保持输出的维度(batch_size, output_size) -> (batch_size, 1)
            # 取输出的最大值的第二个结果（最大值、下标√），得到预测值
            # pred = output.max(dim=1, keepdim=True)[1]
            # print('max:', output.max(dim=1, keepdim=True))
            # print('max[1]:', output.max(dim=1, keepdim=True)[1])
            _, pred = output.max(dim=1, keepdim=True)

            # view_as将target的张量torch.Size([256]) 变成和 pred同样形状的张量torch.Size([256, 1])
            # eq是等于 预测和目标相等
            # 标量求和
            correct += pred.eq(target.view_as(pred)).sum().item()

        # 100×正确除以错误,小数点后保留两位，得到百分比
        percent = '%.2f' % (100 * correct / total)
        # 测试集正确率
        print('Test set: Accuracy {}/{} {}%'.format(correct, total, percent))
    # 返回正确除以总数
    return correct / total

# ------------------------------------------------------------------------------

def time_since(since):
    # 耗时 s秒
    s = time.time()-since
    # 整除 m分钟
    m = math.floor(s/60)
    # 剩余 s秒
    s -= m*60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    # 构建模型
    classifier = RNNClassifier(INPUT_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYER, True)
    # 判断是否需要迁移模型至GPU
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # 开始训练
    print("Training for %d epochs..." % N_EPOCHS)
    start = time.time()
    epoch_list = []
    acc_list = []
    for epoch in range(1, N_EPOCHS+1):
        trainModel()
        acc = testModel()
        epoch_list.append(epoch)
        acc_list.append(acc)

# 循环，起始是1，列表长度+1是终点。步长是1
# epoch = np.arange(1, len(acc_list) + 1, 1)
# 将数据变成一个矩阵
acc_list = np.array(acc_list)
plt.plot(epoch_list, acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

# max: torch.return_types.max(
# values=tensor([[5.1434],
#         [4.4103],
#         [4.1533],
#         [6.9261],
#         [4.0362],
#         [5.6154],
#         [5.1480],
#         [5.4334],
#         [3.5417],
#         [4.9522],
#         [3.9941],
#         [6.4071],
#         [5.2812],
#         [5.3038],
#         [5.0861],
#         [5.4993],
#         [4.5734],
#         [5.0937],
#         [5.1166],
#         [5.0970],
#         [4.7030],
#         [4.4355],
#         [4.2861],
#         [3.0817],
#         [4.0191],
#         [5.0608],
#         [2.8904],
#         [5.8619],
#         [4.9612],
#         [6.4950],
#         [6.7370],
#         [5.1083],
#         [5.0359],
#         [5.2012],
#         [4.0287],
#         [2.8597],
#         [6.8502],
#         [5.1304],
#         [4.9998],
#         [4.1594],
#         [4.5140],
#         [4.9097],
#         [4.6059],
#         [3.4736],
#         [4.1054],
#         [5.1006],
#         [6.9991],
#         [7.0532],
#         [6.3482],
#         [3.8156],
#         [5.3646],
#         [5.6415],
#         [4.2857],
#         [6.9648],
#         [3.8732],
#         [6.3917],
#         [2.9528],
#         [5.1431],
#         [5.4818],
#         [4.4797],
#         [5.8309],
#         [4.9685],
#         [6.0531],
#         [5.0422],
#         [4.7166],
#         [4.2820],
#         [4.3737],
#         [6.8996],
#         [6.9893],
#         [4.9777],
#         [3.7122],
#         [3.7042],
#         [6.9023],
#         [6.7930],
#         [6.7860],
#         [6.9623],
#         [5.3897],
#         [4.9185],
#         [5.8375],
#         [3.3128],
#         [6.0605],
#         [5.7788],
#         [6.9954],
#         [4.9317],
#         [3.3687],
#         [4.0045],
#         [6.1793],
#         [3.6939],
#         [5.3456],
#         [3.2747],
#         [6.9692],
#         [6.6401],
#         [4.2382],
#         [7.0559],
#         [4.8436],
#         [6.3983],
#         [2.5310],
#         [4.7864],
#         [4.4979],
#         [3.9427],
#         [4.8178],
#         [3.9448],
#         [3.6456],
#         [4.6061],
#         [6.2351],
#         [4.1259],
#         [4.1336],
#         [4.5693],
#         [6.5388],
#         [6.7409],
#         [6.6394],
#         [5.4921],
#         [4.3088],
#         [4.1543],
#         [2.8684],
#         [3.9685],
#         [5.7167],
#         [4.7413],
#         [4.7850],
#         [6.6287],
#         [6.9776],
#         [5.7007],
#         [4.6461],
#         [2.8790],
#         [5.2943],
#         [5.0098],
#         [4.8226],
#         [4.1120],
#         [6.1341],
#         [5.1222],
#         [4.1788],
#         [5.5825],
#         [6.7574],
#         [5.0926],
#         [6.4921],
#         [5.2600],
#         [5.7310],
#         [4.5387],
#         [4.3840],
#         [6.9202],
#         [6.7949],
#         [7.0349],
#         [5.0104],
#         [6.8226],
#         [4.6072],
#         [4.7199],
#         [2.8495],
#         [4.7760],
#         [5.6961],
#         [4.8186],
#         [5.7088],
#         [2.3041],
#         [4.5345],
#         [4.4395],
#         [6.2261],
#         [5.1641],
#         [6.9376],
#         [4.6755],
#         [4.3223],
#         [4.2687],
#         [6.9844],
#         [4.7957],
#         [6.4428],
#         [6.6777],
#         [5.9576],
#         [6.6335],
#         [4.0758],
#         [4.4603],
#         [5.9239],
#         [5.2266],
#         [4.8580],
#         [6.7472],
#         [5.7120],
#         [6.6512],
#         [6.8151],
#         [6.8439],
#         [6.8208],
#         [3.4300],
#         [6.8140],
#         [3.5984],
#         [5.3587],
#         [3.4947],
#         [6.8098],
#         [2.3179],
#         [3.8266],
#         [6.5980],
#         [5.6203],
#         [5.6194],
#         [2.1530],
#         [4.1830],
#         [2.5746],
#         [3.9637],
#         [2.3794],
#         [6.5092],
#         [6.3809],
#         [4.0006],
#         [3.5673],
#         [2.7880],
#         [6.7860],
#         [6.7375],
#         [4.9463],
#         [6.5127],
#         [6.3640],
#         [3.8808],
#         [3.1290],
#         [6.9804],
#         [6.9045],
#         [3.3856],
#         [6.3947],
#         [6.7836],
#         [6.6307],
#         [4.8830],
#         [4.6779],
#         [2.6324],
#         [4.7114],
#         [4.5813],
#         [5.0072],
#         [6.6055],
#         [3.7628],
#         [4.7055],
#         [3.1333],
#         [3.4120],
#         [6.5593],
#         [4.6615],
#         [3.9680],
#         [6.5026],
#         [5.4038],
#         [4.1750],
#         [6.8172],
#         [4.4927],
#         [6.4966],
#         [2.4017],
#         [2.4051],
#         [2.3371],
#         [3.8213],
#         [4.6115],
#         [3.9928],
#         [2.7296],
#         [4.3039],
#         [6.5373],
#         [2.7679],
#         [4.9365],
#         [6.3661],
#         [6.4876],
#         [6.5547],
#         [3.9020],
#         [6.3577],
#         [2.3396],
#         [2.4169],
#         [3.3986],
#         [3.3491],
#         [4.0379],
#         [4.1289],
#         [2.9576],
#         [6.1370],
#         [3.2100]], device='cuda:0'),
# indices=tensor([[14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [ 0],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [ 0],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [ 4],
#         [ 4],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [ 0],
#         [ 4],
#         [10],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [14],
#         [ 0],
#         [ 0],
#         [10],
#         [ 0],
#         [14],
#         [14],
#         [ 4],
#         [14],
#         [ 0]], device='cuda:0'))
