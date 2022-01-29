# coding: utf-8

import numpy as np
import cv2
import random
import os

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

# 训练集txt 图片全路径 标签
train_txt_path = os.path.join("..", "..", "Data/train.txt")

CNum = 2000     # 挑选多少图片进行计算

img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    # 读取所有行 所有图片
    lines = f.readlines()
    # 打乱各行 随机挑选图片
    random.shuffle(lines)

    # 挑选CNum=2000张图片 计算均值、标准差
    for i in range(CNum):
        # 每一行 图片的绝对路径
        img_path = lines[i].rstrip().split()[0]
        # 读取一行 作为 一张图片
        img = cv2.imread(img_path)
        # 按照宽高缩放图片 (32, 32, 3)
        img = cv2.resize(img, (img_h, img_w))
        # print('img resize shape: ',img.shape)
        # print('img resize: ', img)

        # np.newaxis的作用是增加一个维度 三维 (32, 32, 3) -> 四维 (32, 32, 3, 1)
        img = img[:, :, :, np.newaxis]
        # print('img newaxis shape: ', img.shape)
        # print('img newaxis: ', img)

        # 按照 axis=3 方向（第四维 batch_size） 拼接每一张图片
        imgs = np.concatenate((imgs, img), axis=3)
        # print('img concatenate shape: ', img.shape)
        # print('img concatenate: ', imgs)
        print(i)

# /255 归一化 像素值归一化至[0-1]之间
imgs = imgs.astype(np.float32)/255.

# RGB flatten
for i in range(3):
    # 以第二维（通道）为基准 所有图片其他维度扁平化拉成一行
    pixels = imgs[:,:,i,:].ravel()
    # 求得当前通道的 平均值
    means.append(np.mean(pixels))
    # 求得当前通道的 标准差
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR， PIL/Skimage 读取到的都是RGB不用转
means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

