import cv2
import common
import numpy as np
import datasets
from sklearn.svm import SVC


def get_feature(img_, hog_, winStride):
    feature_ = hog_.compute(img=img_, winStride=winStride)
    feature_ = np.array(feature_)
    feature_ = feature_.reshape(-1)
    return feature_


if __name__ == "__main__":
    print(common.get_local_time(), "程序运行开始")
    winSize = (32, 32)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    x_train, y_train, x_test, y_test = datasets.load_CIFAR10('data/cifar10/')
    grays = np.array([cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY) for i in range(len(x_train))])
    features = np.array([get_feature(gray, hog, (4, 4)) for gray in grays])
    print(common.get_local_time(), "HOG特征提取完毕")
    new_features = features[:500]
    new_labels = y_train[:500]
    clf = SVC()
    clf.fit(new_features, new_labels)
    print(common.get_local_time(), "SVC训练完成")
    right = 0
    wrong = 0
    for i in range(400, 800):
        gray = cv2.cvtColor(x_test[i], cv2.COLOR_RGB2GRAY)
        feat = get_feature(gray, hog, (4, 4))
        y = clf.predict([feat])
        if y[0] == y_test[i]:
            right += 1
        else:
            wrong += 1
    print(common.get_local_time(), "正确数:", right, ", 错误数:", wrong)
    print(common.get_local_time(), "程序运行结束")
