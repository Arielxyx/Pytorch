# @author: Ariel
# @time: 2021/7/9 13:55

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

# 准备 & 加载数据集
class DiabetesDataSet(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataSet('diabetes.csv.gz')
train_loader = DataLoader(dataset,batch_size=32,shuffle=True,num_workers=2)

# 设计模型类
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# 定义损失函数
criterion = torch.nn.BCELoss(size_average = True)

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader,0):
            inputs, labels = data

            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()