# coding:utf-8
"""
    将cifar10的data_batch_12345 转换成 png格式的图片
    每个类别单独存放在一个文件夹，文件夹名称为0-9

    数据集文件：
        共有 5 个训练集数据文件 data_batch_12345 | 1 个测试集数据文件 test_batch
            每个 data_batch 由10000个 32×32×3大小的图片组成
        故有50000张训练样本 | 10000张测试样本
            每个 data_batch 都是用 cPickle 生成的 Python "pickle"对象
    data:
        10000×3072的numpy数组，数据格式为unit8。
        其中每一行存储了一张32×32的彩色图像。即对于3072个值，每1024个值（32×32=1024）为图片的一个通道数据，一共按顺序包含了红绿蓝三个通道。
    labels:
        长度为10000的数字列表。
        数字的取值范围为0到9之间的整数，表示图片所对应的标签值。
"""

from imageio import imwrite
import numpy as np
import os
import pickle

# 定义 cifar-10原数据集 所在文件夹路径
data_dir = os.path.join("..", "..", "Data", "cifar-10-batches-py")
# 定义 处理后的图片训练集 所在文件夹路径
train_o_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_train")
# 定义 处理后的图片测试集 所在文件夹路径
test_o_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_test")

Train = False   # 不解压训练集，仅解压测试集

# 解压缩，返回解压后的字典
def unpickle(file):
    # rb  只读 （对于不需要进行更新的文件，可以防止用户的错误的写回操作，防止损毁原有数据）
    # rb+ 更新二进制文件 （可以读取，同时也可以写入，需要用到fseek之类的函数进行配合，以免出错）
    with open(file, 'rb') as fo:
        # pickle.load(file，encoding) 把file中的对象读出，encoding 参数可置为 'bytes' 来将这些 8 位字符串实例读取为字节对象。
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

# 创建文件夹
def my_mkdir(my_dir):
    # 如果不是文件夹 则创建
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

# 生成训练集图片，
if __name__ == '__main__':
    # 解压训练集（共有 5 个训练集数据文件 data_batch_12345）
    if Train:
        for j in range(1, 6):
            data_path = os.path.join(data_dir, "data_batch_" + str(j))  # data_batch_12345
            train_data = unpickle(data_path)
            print(data_path + " is loading...")

            for i in range(0, 10000):
                img = np.reshape(train_data[b'data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)

                label_num = str(train_data[b'labels'][i])
                o_dir = os.path.join(train_o_dir, label_num)
                my_mkdir(o_dir)

                img_name = label_num + '_' + str(i + (j - 1)*10000) + '.png'
                img_path = os.path.join(o_dir, img_name)
                imwrite(img_path, img)
            print(data_path + " loaded.")

    # 解压测试集（只有 1 个测试集数据文件 test_batch）
    print("test_batch is loading...")
    # 定义测试集 在哪个文件
    test_data_path = os.path.join(data_dir, "test_batch")
    # 解压缩文件 生成字典 {'filenames': [10000个图片名], 'data': [10000×3072的numpy数组, 'batch_label': 'testing batch 1 of 1', 'labels':[10000个图片标签值] }
    test_data = unpickle(test_data_path)

    # 遍历测试集10000个样本
    for i in range(0, 10000):
        # 10000x3072 numpy数组 的 每一行 -> 每张 3x32x32 的彩色图片
        img = np.reshape(test_data[b'data'][i], (3, 32, 32))
        # 将图片数据格式由（channels,width,height） -> （width,height,channels） 进行格式的转换后方可进行显示
        img = img.transpose(1, 2, 0)
        # 10000x1 numpy数组 的 每一行 -> str类型
        label_num = str(test_data[b'labels'][i])

        # 定义测试集 经处理后的图片 存储在哪个文件夹下
        o_dir = os.path.join(test_o_dir, label_num)
        # 创建文件夹
        my_mkdir(o_dir)

        # 定义图片名：所属标签_第几张图片.png
        img_name = label_num + '_' + str(i) + '.png'

        # 定义图片全路径名：文件夹路径/图片名
        img_path = os.path.join(o_dir, img_name)

        # 将处理后的图片 写入指定图片路径
        imwrite(img_path, img)

    print("test_batch loaded.")
