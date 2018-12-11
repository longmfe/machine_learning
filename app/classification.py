# encoding: utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from lib.classification.get_mnist_data import get_mnist_data

def load_data():
    get_data = get_mnist_data()
    train_images = get_data.load_train_images()
    train_labels = get_data.load_train_labels()
    test_images = get_data.load_test_images()
    test_labels = get_data.load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print('done')
    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    train_img, train_lbl, test_img, test_lbl = load_data()
    print("train_img: ", train_img)
    print("train_lbl: ", train_lbl)
    print("test_img: ", test_img)
    print("test_img: ", test_lbl)

