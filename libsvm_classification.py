# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:55:23 2024

@author: 87712
"""

# libsvm_classification.py

import numpy as np
from libsvm.svmutil import svm_train, svm_predict, svm_problem, svm_parameter
import joblib

# 加载特征和标签
hog_train_features = np.load('hog_train_features.npy')
hog_test_features = np.load('hog_test_features.npy')
sift_train_features = np.load('sift_train_features.npy')
sift_test_features = np.load('sift_test_features.npy')
color_hist_train_features = np.load('color_hist_train_features.npy')
color_hist_test_features = np.load('color_hist_test_features.npy')
pixel_train_features = np.load('pixel_train_features.npy')
pixel_test_features = np.load('pixel_test_features.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')
scaler = joblib.load('scaler.pkl')

# LibSVM分类
def libsvm_classify(train_features, train_labels, test_features, test_labels, kernel_type=0):
    problem = svm_problem(train_labels.tolist(), train_features.tolist())
    param = svm_parameter(f'-t {kernel_type} -c 1')
    model = svm_train(problem, param)
    p_label, p_acc, p_val = svm_predict(test_labels.tolist(), test_features.tolist(), model)
    print(f"LibSVM Classification Report with kernel type {kernel_type}:\n", p_acc)

# 组合测试
features = {
    'HOG': (hog_train_features, hog_test_features),
    'SIFT': (sift_train_features, sift_test_features),
    'Color Histogram': (color_hist_train_features, color_hist_test_features),
    'Pixel': (pixel_train_features, pixel_test_features)
}

kernels = {
    'linear': 0,
    'rbf': 2
}

for feature_name, (train_features, test_features) in features.items():
    for kernel_name, kernel_type in kernels.items():
        print(f"\n{feature_name} features with {kernel_name} kernel LibSVM:")
        libsvm_classify(train_features, train_labels, test_features, test_labels, kernel_type=kernel_type)
