# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 100
max_iterations = 10000


optimizer = Adam()

network = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch,t_batch)
    optimizer.update(network.params, grad)
    loss = network.loss(x_batch, t_batch)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


y = network.predict(x_test) # x에 대해 예측
y = np.argmax(y, axis=1) # 그중 가장 크게 나온 값의 인덱스
t_answer = np.argmax(t_test, axis=1) # 실제 타겟값의 가장 큰 인덱스(one-hot인코딩이므로)

wrong_answers = y != t_answer # 틀린 지점들의 인덱스 가져옴
wrong_imgs = x_test[wrong_answers] # 틀린지점의 이미지 뽑아내기

y_data = y[wrong_answers] # 틀린지점 y값
t_data = t_answer[wrong_answers] # 틀린지점 정답 값

print('the number of wrong answers:',len(y_data)) # 틀린 갯수
for i in range(len(y_data)):
    print('network:',y_data[i],'real:',t_data[i]) # 신경망의 예측과 실제 값 출력

wrong_imgs = wrong_imgs * 255 # 정규화된 이미지를 다시 원상복귀 시킨다.
wrong_imgs = wrong_imgs.reshape(-1,28) # 세로로 출력되게끔 설정
img_show(wrong_imgs) # 이미지 출력