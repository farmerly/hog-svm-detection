import pickle
import os
import cv2
import numpy as np

lableName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def pretreatment(pathname_):
    dst_imgs_ = []
    imglist_ = [cv2.imread(os.path.join(pathname_, file_)) for file_ in os.listdir(pathname_)]
    for img_ in imglist_:
        row_, col_, chn_ = img_.shape
        # dst_row_ = row_ if row_ % 8 == 8 else row_ - row_ % 8
        # dst_col_ = col_ if col_ % 8 == 8 else col_ - col_ % 8
        # if row_ != dst_row_ or col_ != dst_col_:
        #     dst_imgs_.append(cv2.resize(img_, (dst_col_, dst_row_)))
        # else:
        #     dst_imgs_.append(img_)
        dst_imgs_.append(cv2.resize(img_, (32, 32)))
    return dst_imgs_


def load_CIFAR_batch(filename):
    """
    cifar-10数据集是分batch存储的，这是载入单个batch
    @参数 filename: cifar文件名
    @r返回值: X, Y: cifar batch中的 data 和 labels
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """
    读取载入整个 CIFAR-10 数据集
    @参数 ROOT: 根目录名
    @return: X_train, Y_train: 训练集 data 和 labels
             X_test, Y_test: 测试集 data 和 labels
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_CIFAR10('data/cifar-10-batches-py/')
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
