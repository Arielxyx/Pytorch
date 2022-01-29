# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集
    在研究过程中：
        验证集和测试集作用相同：只是对模型进行一个观测，观测训练好的模型的泛化能力。
    在工程应用中：
        验证集：从训练集里再划分出来的一部分作为验证集，用来选择模型和调参的。
        测试集：当调好之后，再用测试集对该模型进行泛化性能的评估，如果性能OK，再把测试集输入到模型中训练，最终得到的模型就是提交给用户的模型。
"""

import os
import glob
import random
import shutil

# 定义 处理后的图片测试集 所在文件夹路径
dataset_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_test")
# 定义 训练集 所在文件夹路径
train_dir = os.path.join("..", "..", "Data", "train")
# 定义 验证集 所在文件夹路径
valid_dir = os.path.join("..", "..", "Data", "valid")
# 定义 测试集 所在文件夹路径
test_dir = os.path.join("..", "..", "Data", "test")

# 训练集 占比 80%
train_per = 0.8
# 验证集 占比 80%
valid_per = 0.1
# 测试集 占比 80%
test_per = 0.1

# 创建文件夹
def makedir(new_dir):
    # 如果文件夹不存在 则创建
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

if __name__ == '__main__':
    # root	表示正在遍历的文件夹本身  （根/子） G:\Pytorch\PyTorch_Tutorial - master\Data\cifar - 10 - png\raw_test（还会继续遍历子文件夹 0 1 2 3 4 5 6 7 8 9）
    # dirs	记录正在遍历的文件夹下的 子文件夹集合 0 1 2 3 4 5 6 7 8 9
    # files	记录正在遍历的文件夹中的 文件集合
    for root, dirs, files in os.walk(dataset_dir):
        print('root: ', root)
        # 遍历子文件夹 0 1 2 3 4 5 6 7 8 9
        for sDir in dirs:
            print('sDir: ', sDir)
            # glob.glob一次性读取 对应文件夹下所有符合要求的 子文件夹、子文件夹下的文件 列表
            # 常见的两个方法有glob.glob()和glob.iglob()，可以和常用的find功能进行类比，glob支持*?[]这三种通配符
            imgs_list = glob.glob(os.path.join(root, sDir, '*.png'))

            # 当seed()没参数时，每次生成的随机数是不一样的
            # 当seed()有参数时，每次生成的随机数是一样的
            # 同时选择不同的参数，生成的随机数也不一样
            random.seed(666)
            # 当random.seed()设定一个初始值时，random.shuffle()的顺序保持不变
            random.shuffle(imgs_list)

            # 当前子文件夹的 样本量
            imgs_num = len(imgs_list)
            # 训练集 截至 图片数量
            train_point = int(imgs_num * train_per)
            # 验证集 截至 图片数量
            valid_point = int(imgs_num * (train_per + valid_per))

            for i in range(imgs_num):
                # 训练集 [0, train_point)
                if i < train_point:
                    out_dir = os.path.join(train_dir, sDir)
                # 验证集 [train_point, valid_point)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sDir)
                # 测试集 [valid_point, imgs_num]
                else:
                    out_dir = os.path.join(test_dir, sDir)

                # 创建文件夹
                makedir(out_dir)
                # 定义 目标文件夹路径
                # os.path.split()：按照路径将文件名和路径分割开 ..\..\Data\cifar-10-png\raw_test\1\1_7986.png
                out_path = os.path.join(out_dir, os.path.split(imgs_list[i])[-1])
                # 功能：复制文件 格式：shutil.copy('来源文件','目标地址')
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sDir, train_point, valid_point-train_point, imgs_num-valid_point))
