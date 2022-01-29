# coding: utf-8
import os
import torch
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from Code.utils.utils import MyDataset, Net, normalize_invert
from torch.utils.data import DataLoader

vis_layer = 'conv1'
log_dir = os.path.join("..", "..", "Result", "visual_featuremaps")
# 仅一张图片 b=1
txt_path = os.path.join("..", "..", "Data", "visual.txt")
pretrained_path = os.path.join("..", "..", "Data", "net_params_72p.pkl")

# 构建新模型 并将旧模型的参数迁移过来
net = Net()
pretrained_dict = torch.load(pretrained_path)
net.load_state_dict(pretrained_dict)

# 数据预处理
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)
testTransform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normTransform
])
# 载入数据
test_data = MyDataset(txt_path=txt_path, transform=testTransform)
test_loader = DataLoader(dataset=test_data, batch_size=1)
img, label = iter(test_loader).next()
# [1, 3, 32, 32]
x = img
# print("img:",img.size())

writer = SummaryWriter(log_dir=log_dir)

# 遍历模型的每一层网络
for name, layer in net._modules.items():
    # 为fc层预处理x [b, -1]
    x = x.view(x.size(0), -1) if "fc" in name else x

    # 对x执行单层运算
    x = layer(x)
    print(x.size())

    # 由于__init__()相较于forward()缺少relu操作，需要手动增加
    x = F.relu(x) if 'conv' in name else x
    # print('relu: ',x.size())

    # 依据选择的层（这里选择的是卷积核conv1），进行记录feature maps
    if name == vis_layer:
        # 绘制feature maps
        x1 = x.transpose(0, 1)  # B，C, H, W [1, 6, 28, 28] ---> [6, 1, 28, 28]
        # print('transpose: ',x1.size())
        img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=3) # conv1: C/3=6/3=2行 源代码nrow=2
        writer.add_image(vis_layer + '_feature_maps', img_grid, global_step=666)

        # 绘制原始图像
        img_raw = normalize_invert(img, normMean, normStd)  # 原始图像不需要标准化 -> 去标准化
        # np.clip() 裁剪（限制）数组中的值 即整个数组的值限制在指定值a_min,与a_max之间，对比a_min小的和比a_max大的值就重置为a_min,和a_max。
        # squeeze() 去掉 [指定的] 维数为一的维度
        # uint8类型取值范围：0到255
        img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8') # 原始图像不需要归一化 -> 去归一化
        writer.add_image('raw img', img_raw, global_step=666)  # j 表示feature map数
writer.close()