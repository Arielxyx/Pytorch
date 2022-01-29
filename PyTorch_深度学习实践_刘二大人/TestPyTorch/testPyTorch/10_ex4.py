# @author: Ariel
# @time: 2021/7/12 7:46

### 下采样 使用其中的一种方法：最大池化层 maxpooling

import torch

input = [3,4,6,5,
         2,4,6,8,
         1,6,7,8,
         9,7,4,6]
input = torch.Tensor(input).view(1,1,4,4)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)
print(output.shape) # ([1, 1, 2, 2])
print(output.data)