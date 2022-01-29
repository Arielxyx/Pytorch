# coding: utf-8
import os
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

resnet18 = models.resnet18(False)
writer = SummaryWriter(os.path.join("..", "..", "Result", "runs"))
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]


for n_iter in range(100):
    # --------------------- add_scalar(title, y, x, walltime) 在一个图表中记录一个标量的变化 --------------------------------------
    # torch.rand(*sizes, out=None) → Tensor 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，张量的形状由参数sizes定义。
    s1 = torch.rand(1)  # value to keep
    s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar(os.path.join("data", "scalar_systemtime"), s1[0], n_iter)
    # data grouping by `slash`
    writer.add_scalar(os.path.join("data", "scalar_customtime"), s1[0], n_iter, walltime=n_iter)

    # --------------------- add_scalars(title, {s_key:s_value}, x, walltime) 在一个图表中记录多个标量的变化 --------------------------------------
    writer.add_scalars(os.path.join("data", "scalar_group"),
                       {"xsinx": n_iter * np.sin(n_iter),
                        "xcosx": n_iter * np.cos(n_iter),
                        "arctanx": np.arctan(n_iter)},
                       n_iter)

    # --------------------- add_image(title, image, x, walltime) 绘制图片 --------------------------------------
    x = torch.rand(32, 3, 64, 64)  # output from network
    if n_iter % 10 == 0:
        # 将一组图片拼接成一张图片 便于可视化
        x = vutils.make_grid(x, normalize=True, scale_each=True)
        # print('add_image:',x.size())
        writer.add_image('Image', x, n_iter)  # Tensor
        # writer.add_image('astronaut', skimage.data.astronaut(), n_iter) # numpy
        # writer.add_image('imread',
        # skimage.io.imread('screenshots/audio.png'), n_iter) # numpy

        # --------------------- add_audio(title, value, x) 记录音频 --------------------------------------
        x = torch.zeros(sample_rate * 2)
        for i in range(x.size(0)):
            # sound amplitude should in [-1, 1]
            x[i] = np.cos(freqs[n_iter // 10] * np.pi *
                          float(i) / float(sample_rate))
        writer.add_audio('myAudio', x, n_iter)

        # --------------------- add_text(title, value, x) 记录文字 --------------------------------------
        writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
        writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)
        print('epoch =',str(n_iter))

        # --------------------- add_histogram(tag=name, values=param, y, bins, walltime) 绘制直方图和多分位数折线图 --------------------------------------
        for name, param in resnet18.named_parameters():
            if 'bn' not in name:
                # print('name',name)
                # print('param',param)
                writer.add_histogram(name, param, n_iter)

        # --------------------- add_pr_curve(title, labels, predictions) 绘制 PR 曲线 --------------------------------------
        # np.random.randint(2, size=100) 100个[0,2)的整数 | np.random.rand(100) 100个[0,1)的整数
        writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)  # needs tensorboard 0.4RC or later
        writer.add_pr_curve_raw('prcurve with raw data',
                                true_positive_counts, # TP
                                false_positive_counts, # FP
                                true_negative_counts, # TN
                                false_negative_counts, # FN
                                precision,
                                recall, n_iter)

# --------------------- export_scalars_to_json(path) 将scalar信息保存到json文件 --------------------------------------
writer.export_scalars_to_json(os.path.join("..", "..", "Result", "all_scalars.json"))

# --------------------- add_embedding(mat=features, metadata=images, label_img=images) 在三维空间或二维空间展示数据分布 --------------------------------------
# 仅测试集
dataset = datasets.MNIST(os.path.join("..", "..", "Data", "mnist"), train=False, download=True)
# print('dataset.test_data.size',dataset.test_data.size()) # [10000, 28, 28]
images = dataset.test_data[:500].float()
# print('images.size',images.size()) # [100, 28, 28]
label = dataset.test_labels[:500]
# print('label.size',label.size()) # [100]
features = images.view(500, 784)

writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
writer.add_embedding(features, global_step=1, tag='noMetadata')

# 全部数据集 测试集 + 训练集
dataset = datasets.MNIST(os.path.join("..", "..", "Data", "mnist"), train=True, download=True)
images_train = dataset.train_data[:500].float()
labels_train = dataset.train_labels[:500]
features_train = images_train.view(500, 784)

all_features = torch.cat((features, features_train))
all_labels = torch.cat((label, labels_train))
all_images = torch.cat((images, images_train))
dataset_label = ['test'] * 500 + ['train'] * 500
all_labels = list(zip(all_labels, dataset_label))

writer.add_embedding(all_features, metadata=all_labels, label_img=all_images.unsqueeze(1),
                     metadata_header=['digit', 'dataset'], global_step=2)

# --------------------- add_video(title, value, fps) 记录视频 --------------------------------------
vid_images = dataset.train_data[:16 * 48]
vid = vid_images.view(16, 1, 48, 28, 28)  # BxCxTxHxW
writer.add_video('video', vid_tensor=vid)
writer.add_video('video_1_fps', vid_tensor=vid, fps=1)

# # --------------------- add_graph() 绘制网络结构拓扑图 --------------------------------------
# dummy_input = torch.rand(6, 3, 224, 224)
# writer.add_graph(resnet18, dummy_input)

writer.close()