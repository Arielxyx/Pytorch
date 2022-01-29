# coding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt



def show_confMat(confusion_mat, classes_name, set_name, out_dir):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :param set_name: str, eg: 'valid', 'train'
    :param out_dir: str, png输出的文件夹
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    # 遍历 全部类别个数 这么多次
    for i in range(len(classes_name)):
        # confusion_mat_N[i, :] 第i行 所有列
        # 处理第i行 = 均 ÷ ∑当前行每列值
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar() # 添加指示色标 默认垂直

    # 设置文字
    xlocations = np.array(range(len(classes_name))) # 刻度标记 0 1 2 3 4 5 6 7 8 9（len(classes_name=10)）
    plt.xticks(xlocations, classes_name, rotation=60) # x轴 每个刻度对应的文字 旋转60°
    plt.yticks(xlocations, classes_name) # y轴 每个刻度对应的文字
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]): # shape[0] = len(classes_name) = 10
        for j in range(confusion_mat_N.shape[1]): # shape[1] = len(classes_name) = 10
            # 打印位置在混淆矩阵(i,j)上的数字confusion_mat[i, j] 位于中间（vertical垂直居中、horizontal水平居中） 红色 大小=10
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'))
    plt.close()

if __name__ == '__main__':

    print('QQ group: {}, password: {}'.format(671103375, 2018))

