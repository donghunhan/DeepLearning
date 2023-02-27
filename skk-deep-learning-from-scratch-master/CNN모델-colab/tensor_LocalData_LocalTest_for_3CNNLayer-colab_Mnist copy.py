import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist

# Local의 dataset folder의 Mnist 데이터 읽기
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=False,one_hot_label=False)

model = load_model('mnist_cnn_model.h5') # model과 weights load

#(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

train_loss, train_acc = model.evaluate(train_images,  train_labels, verbose=2)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('train:',train_acc)
print('test:',test_acc)