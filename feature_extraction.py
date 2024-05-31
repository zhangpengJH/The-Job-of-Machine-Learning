# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:19:11 2024

@author: 87712
"""

# feature_extraction.py

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from skimage.feature import hog
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import joblib

# 数据准备
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# 特征提取
def extract_hog_features(images):
    hog_features = []
    for img in images:
        hog_feature = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(hog_feature)
    return np.array(hog_features)

def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors_list.append(des.astype(np.float32))  # 确保数据类型为float32
        else:
            descriptors_list.append(np.zeros((1, 128), dtype=np.float32))  # 如果没有特征点，添加一个零向量
    return descriptors_list

def compute_bow_features(descriptors_list, k=100):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
    all_descriptors = np.vstack(descriptors_list).astype(np.float32)  # 确保数据类型为float32
    kmeans.fit(all_descriptors)
    bow_features = []
    for descriptors in descriptors_list:
        histogram = np.bincount(kmeans.predict(descriptors), minlength=k)
        bow_features.append(histogram)
    return np.array(bow_features)

def extract_color_histogram_features(images, bins=32):
    hist_features = []
    for img in images:
        hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
        hist_features.append(hist)
    return np.array(hist_features)

# 转换为3通道图像以适应SIFT和颜色直方图
train_images_color = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for img in train_images]
test_images_color = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for img in test_images]

# 提取HOG特征
hog_train_features = extract_hog_features(train_images)
hog_test_features = extract_hog_features(test_images)

# 提取SIFT特征并计算词袋模型特征
train_sift_descriptors = extract_sift_features(train_images_color)
test_sift_descriptors = extract_sift_features(test_images_color)
sift_train_features = compute_bow_features(train_sift_descriptors)
sift_test_features = compute_bow_features(test_sift_descriptors)

# 提取颜色直方图特征
color_hist_train_features = extract_color_histogram_features(train_images_color)
color_hist_test_features = extract_color_histogram_features(test_images_color)

# 提取像素值特征
pixel_train_features = train_images.reshape((train_images.shape[0], -1))
pixel_test_features = test_images.reshape((test_images.shape[0], -1))

# 特征标准化
scaler = StandardScaler()
hog_train_features = scaler.fit_transform(hog_train_features)
hog_test_features = scaler.transform(hog_test_features)
sift_train_features = scaler.fit_transform(sift_train_features)
sift_test_features = scaler.transform(sift_test_features)
color_hist_train_features = scaler.fit_transform(color_hist_train_features)
color_hist_test_features = scaler.transform(color_hist_test_features)
pixel_train_features = scaler.fit_transform(pixel_train_features)
pixel_test_features = scaler.transform(pixel_test_features)

# 保存特征和标签
np.save('hog_train_features.npy', hog_train_features)
np.save('hog_test_features.npy', hog_test_features)
np.save('sift_train_features.npy', sift_train_features)
np.save('sift_test_features.npy', sift_test_features)
np.save('color_hist_train_features.npy', color_hist_train_features)
np.save('color_hist_test_features.npy', color_hist_test_features)
np.save('pixel_train_features.npy', pixel_train_features)
np.save('pixel_test_features.npy', pixel_test_features)
np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)
joblib.dump(scaler, 'scaler.pkl')
