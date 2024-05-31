# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:35:32 2024

@author: 87712
"""

# svm_classification.py

# svm_classification.py

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
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

# SVM分类
def svm_classify(train_features, train_labels, test_features, test_labels, kernel='linear'):
    clf = svm.SVC(kernel=kernel)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    report = classification_report(test_labels, predictions)
    print(f"SVM Classification Report with {kernel} kernel:\n", report)

# 组合测试
features = {
    'HOG': (hog_train_features, hog_test_features),
    'SIFT': (sift_train_features, sift_test_features),
    'Color Histogram': (color_hist_train_features, color_hist_test_features),
    'Pixel': (pixel_train_features, pixel_test_features)
}

kernels = {
    'linear': 'linear',
    'rbf': 'rbf'
}

for feature_name, (train_features, test_features) in features.items():
    for kernel_name in kernels.values():
        print(f"\n{feature_name} features with {kernel_name} kernel SVM:")
        svm_classify(train_features, train_labels, test_features, test_labels, kernel=kernel_name)
