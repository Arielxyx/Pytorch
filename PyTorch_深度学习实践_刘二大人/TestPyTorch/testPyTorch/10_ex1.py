# @author: Ariel
# @time: 2021/7/12 7:10

### 定义卷积结构 输出 输入层、卷积层、输出层的形状大小

import torch

batch_size = 1 # B - 每个批次的样本数 此处默认一个批次 故为总样本数
width, height = 100, 100 # w、h - 图像的宽、高

in_channels, out_channels = 5, 10 # n、m - 输入、输出 通道数
kernel_size = 3 # k - 卷积核大小

input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)

conv_layer = torch.nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size = kernel_size)

output = conv_layer(input)

# 输入层：B c=n w h ([1, 5, 100, 100])
print(input.shape)
# 卷积层：n m k1 k2 ([10, 5, 3, 3])
print(conv_layer.weight.shape)
# 输出层：B m w-w`/2*2 h-h`/2*2 ([1, 10, 98, 98])
print(output.shape)