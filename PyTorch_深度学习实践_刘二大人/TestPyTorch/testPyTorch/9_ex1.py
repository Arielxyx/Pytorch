# @author: Ariel
# @time: 2021/7/10 17:18

### NLLLoss
import numpy as np

# 实际标签值
y = np.array([1, 0, 0])
# 预测值 1*3
z = np.array([0.2, 0.1, -0.1])

# softmax处理（解决多分类）
y_pred = np.exp(z)/np.exp(z).sum()
print(np.exp(z)) # [e^z1 e^z2 e^z3]
print(np.exp(z).sum()) # e^z1 + e^z2 + e^z3
print(y_pred)

# 损失值 NLLLoss
loss = (-y*np.log(y_pred)).sum()
print(np.log(y_pred)) # logy_pred
print(-y*np.log(y_pred)) # -y * logy_pred 两个矩阵维度均为3*1 一一对应相乘
print(loss)
