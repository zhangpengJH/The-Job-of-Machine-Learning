# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:59:24 2024

@author: 87712
"""

# nn_classification.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
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

# 前馈神经网络分类
def neural_network_classify(train_features, train_labels, test_features, test_labels):
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(train_features.shape[1],)))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_features, train_labels, epochs=10, batch_size=200, verbose=2)
    
    scores = model.evaluate(test_features, test_labels, verbose=0)
    print(f"Neural Network Accuracy: {scores[1] * 100}")

# 组合测试
features = {
    'HOG': (hog_train_features, hog_test_features),
    'SIFT': (sift_train_features, sift_test_features),
    'Color Histogram': (color_hist_train_features, color_hist_test_features),
    'Pixel': (pixel_train_features, pixel_test_features)
}

for feature_name, (train_features, test_features) in features.items():
    print(f"\nNeural Network with {feature_name} features:")
    neural_network_classify(train_features, train_labels, test_features, test_labels)
