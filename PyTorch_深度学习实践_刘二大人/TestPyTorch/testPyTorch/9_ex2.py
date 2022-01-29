# @author: Ariel
# @time: 2021/7/10 17:34

### CrossEntropyLoss batch-size = 1
import torch

# 实际标签值
y = torch.LongTensor([0])
print(y)
# 预测值 1*3
z = torch.Tensor([[0.2, 0.1, -0.1]])
print(z)

# 损失值 CrossEntropyLoss （包含softmax处理）
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z,y) # 这里的z为最原始的预测值 无需经过softmax处理为y_pred
print(loss)
print(loss.item())