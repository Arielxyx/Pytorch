# coding:utf-8
import os
'''
    为数据集生成对应的txt文件
'''

# 训练集 将图片路径、标签信息存储在一个 txt
train_txt_path = os.path.join("..", "..", "Data", "train.txt")
# 训练集 图片所在文件夹
train_dir = os.path.join("..", "..", "Data", "train")

# 验证集 将图片路径、标签信息存储在一个 txt
valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")
# 验证集 图片所在文件夹
valid_dir = os.path.join("..", "..", "Data", "valid")


def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    # 遍历 train 文件夹下各文件夹
    # topdown 为真，则优先遍历top目录，否则优先遍历top的子目录(默认为开启)
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):
        # 遍历 各分类子文件夹
        for sub_dir in s_dirs:
            # 获取 各分类子文件夹 绝对路径
            i_dir = os.path.join(root, sub_dir)
            # os.listdir() 方法用于返回指定的文件夹 包含的文件 or 文件夹的 名字列表。
            img_list = os.listdir(i_dir)

            # 遍历 各分类子文件夹 所有图片
            for i in range(len(img_list)):
                # 若不是png文件 跳过
                if not img_list[i].endswith('png'):
                    continue
                # 图片 所属类别
                label = img_list[i].split('_')[0]
                # 图片 绝对路径
                img_path = os.path.join(i_dir, img_list[i])
                # 待写入 每行内容：图片绝对路径 所属类别
                line = img_path + ' ' + label + '\n'
                # 写入 txt
                f.write(line)
    f.close()


if __name__ == '__main__':
    # 生成 训练集 txt
    gen_txt(train_txt_path, train_dir)
    # 生成 验证集 txt
    gen_txt(valid_txt_path, valid_dir)

