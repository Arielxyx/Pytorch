# @author: Ariel
# @time: 2021/3/25 10:11

import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w=1.0

def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

def gradient(x,y):
    return 2*x*(forward(x)-y)

epoch_list = []
loss_list = []

print('Predict（before training）', 4, forward(4))
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        loss_val = loss(x_val, y_val)
        grad_val = gradient(x_val, y_val)
        w -= 0.01*grad_val

    epoch_list.append(epoch)
    loss_list.append(loss_val)
    print('epoch :', epoch, '| w =', w, '| loss =', loss_val)

print('Predict（after training）', 4, forward(4))

plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.grid(ls='--')
plt.show()

