import math
import numpy as np


def rgb2gray(img_):
    """
    RGB图像转灰度图像
    :param img_: 原始图片
    :return:
    """
    return img_[:, :, 0] * 0.2989 + img_[:, :, 1] * 0.5870 + img_[:, :, 2] * 0.1140


def gradientGray(gray_, row_, col_):
    """
    :param gray_: 灰度图像
    :param row_: X方向长度
    :param col_: Y方向长度
    :return:
    """
    gray_f = np.copy(gray_).astype("float")
    _gradient = np.zeros((row_, col_))
    _theta = np.zeros((row_, col_))
    for x in range(row_ - 1):
        for y in range(col_ - 1):
            gx = gray_f[x + 1, y] - gray_f[x, y]
            gy = gray_f[x, y + 1] - gray_f[x, y]
            # 梯度幅值, sqrt(gx * gx + gy * gy)
            _gradient[x, y] = math.sqrt(gx * gx + gy * gy)
            # 梯度方向, arctan(gx/gy) * 180 / PI
            v = math.atan2(gx, gy) * 180 / np.pi
            _theta[x, y] = v + 180 if v < 0 else v
    return _gradient.astype("uint8"), _theta.astype("uint8")


def descriptor(gray_, row_, col_, cellSize_=(8, 8), blockSize_=(16, 16), blockStride_=(8, 8)):
    """
    得到hog描述子
    :param gray_: 灰度图像
    :param row_: 宽度
    :param col_: 高度
    :param cellSize_: 细胞单元大小, 默认(8,8)
    :param blockSize_: 块大小, 默认(16,16)
    :param blockStride_: 块步长
    :return:
    """
    gradient, theta = gradientGray(gray_, row_, col_)
    cell_w_count = int(row_ / cellSize_[0])
    cell_h_count = int(col_ / cellSize_[1])
    cell_vector = np.zeros((cell_w_count, cell_h_count, 9))
    for x in range(cell_w_count):
        for y in range(cell_h_count):
            cell_n = np.zeros(9)
            for i in range(cellSize_[0] * x, cellSize_[0] * (x + 1)):
                for j in range(cellSize_[1] * y, cellSize_[1] * (y + 1)):
                    if theta[i, j] % 20 == 0:
                        cell_n[int(theta[i, j] / 20 % 9)] += gradient[i, j]
                    else:
                        rate = theta[i, j] % 20 / 20.0
                        cell_n[math.floor(theta[i, j] / 20)] += gradient[i, j] * rate
                        cell_n[math.ceil(theta[i, j] / 20) % 9] += gradient[i, j] * (1 - rate)
            cell_vector[x, y] = cell_n
    # 块个数
    block_w_count = int((row_ - blockSize_[0]) / blockStride_[0] + 1)
    block_h_count = int((col_ - blockSize_[1]) / blockStride_[1] + 1)
    block_vector = np.zeros((block_w_count, block_h_count, 9))
    for m in range(block_w_count):
        for n in range(block_h_count):
            print("block", m, n)
            for i in range(m, int(m + blockSize_[0] / cellSize_[0])):
                for j in range(n, int(n + blockSize_[1] / cellSize_[1])):
                    print("cell", i, j)
