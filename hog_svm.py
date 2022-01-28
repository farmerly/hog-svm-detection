import os
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


def pretreatment(pathname_):
    imglist_ = [cv2.imread(os.path.join(pathname_, file_)) for file_ in os.listdir(pathname_)]
    for img_ in imglist_:
        row_, col_, chn_ = img_.shape
        print(row_, col_)


if __name__ == "__main__":
    print(common.get_local_time(), "程序运行开始")
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    x_train, y_train, x_test, y_test = datasets.load_CIFAR10('data/cifar-10-batches-py/')
    # 获取灰度图
    grays = np.array([cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY) for i in range(len(x_train))])
    # 获取矢量数据
    features = np.array([get_feature(gray, hog, (8, 8)) for gray in grays])
    print(common.get_local_time(), "HOG特征提取完毕")
    new_features = features[:2000]
    new_labels = y_train[:2000]
    clf = SVC()
    # 开始SVC训练
    print(common.get_local_time(), "开始SVC训练")
    clf.fit(new_features, new_labels)
    print(common.get_local_time(), "SVC训练完成")
    right = 0
    wrong = 0

    for i in range(len(x_test[:200])):
        gray = cv2.cvtColor(x_test[i], cv2.COLOR_RGB2GRAY)
        feat = get_feature(gray, hog, (8, 8))
        y = clf.predict([feat])
        # pic = cv2.resize(x_test[i], (128, 128))
        # cv2.imshow(datasets.lableName[y[0]], pic)
        # cv2.waitKey(0)
        if y[0] == y_test[i]:
            right += 1
        else:
            wrong += 1
    cv2.destroyAllWindows()
    print(common.get_local_time(), "正确数:", right, ", 错误数:", wrong)
    print(common.get_local_time(), "程序运行结束")
