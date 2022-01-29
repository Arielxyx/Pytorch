# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        # 遍历 txt 文件的每一行
        for line in fh:
            # rstrip() 删除 string 字符串末尾的指定字符（默认为空格）.
            line = line.rstrip()
            # 以空格为分隔符： 图片全路径 所属类别
            words = line.split()
            imgs.append((words[0], int(words[1])))

        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # Image.open 对图片进行读取，img 类型为 Image，mode=‘RGB’
        # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img = Image.open(fn).convert('RGB')

        # 在这里做transform，转为tensor等等
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)