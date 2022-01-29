# @author: Ariel
# @time: 2021/3/24 11:26

import numpy as np
import matplotlib.pyplot as plt
# 绘制3D坐标的函数
# https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html#surface-plots
from mpl_toolkits.mplot3d import Axes3D
# 解决坐标轴不能显示中文问题
# http://www.360doc.com/content/14/0713/12/16740871_394080556.shtml
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 准备数据集
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]

# 定义模型 y = wx + b
def forward(x):
    # w、b都是二维矩阵，x * w + b会遵循矩阵计算规则，依次计算每个预测值（z轴上的点），return的结果也是个二维矩阵
    return x * w + b

# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    # 同理：也是个二维矩阵
    return (y_pred-y) * (y_pred-y)

# 权重 W 从0.0到4.0 间隔0.1取数
W = np.arange(0.0, 4.1, 0.1)
# 偏置 B 从0.0到4.0 间隔0.1取数
B = np.arange(0.0, 4.1, 0.1)
# 函数用两个坐标轴上的点在平面上画网格 https://blog.csdn.net/lllxxq141592654/article/details/81532855
# w、b都是二维坐标矩阵（二维数组），两个矩阵的对应元素组成一个坐标点
[w,b] = np.meshgrid(W,B)
print('w\n', w)
print('b\n', b)

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val) # y_pred_val是个二维矩阵
    loss_val = loss(x_val, y_val) # loss_val也是个二维矩阵
    l_sum += loss_val
    print('\nx_val',x_val)
    print('y_val',y_val)
    print('y_pred_val\n', y_pred_val)
    print('loss_val\n', loss_val)
print('\nmse\n', l_sum/3)

# 创建一个绘图对象
fig = plt.figure()
# 用这个绘图对象创建一个Axes对象 - 有3D坐标
ax = Axes3D(fig)
# 用取样点(x,y,z)去构建曲面
ax.plot_surface(w, b, l_sum/3, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.set_xlabel('权重 w', color='r')
ax.set_ylabel('偏置项 b', color='g')
ax.set_zlabel('损失值 mse', color='b')#给三个坐标轴注明
# plt.show() # 显示模块中的所有绘图对象
# tight_layout在plt.savefig的调用方式相对比较稳定，我们将plt.show()函数替换为plt.savefig函数，替换后会在本地另外为png图片，该图片中子图填充了整个图像区域
# https://www.imooc.com/article/296147
plt.savefig('fig.png',bbox_inches='tight')