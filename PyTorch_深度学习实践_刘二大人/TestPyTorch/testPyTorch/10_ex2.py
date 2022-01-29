# @author: Ariel
# @time: 2021/7/12 7:36

### padding = 1
import torch

input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]
input = torch.Tensor(input).view(1,1,5,5)

con_layer = torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
kernal = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
con_layer.weight.data = kernal.data

output = con_layer(input)
print(output.shape) # ([1, 1, 5, 5])
print(output.data)