import cv2
import numpy as np
import datasets


def get_feature(img_, hog_, winStride):
    feature = hog_.compute(img=img_, winStride=winStride)
    feature = np.array(feature)
    feature = feature.reshape(-1)
    return feature


if __name__ == "__main__":
    winSize = (32, 32)
    blockSize = (4, 4)
    blockStride = (2, 2)
    cellSize = (2, 2)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    x_train, y_train, x_test, y_test = datasets.load_CIFAR10('data/cifar10/')
    grays = np.array([cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY) for i in range(len(x_train))])
    features = np.array([get_feature(gray, hog, (2, 2)) for gray in grays])
    print(features)
