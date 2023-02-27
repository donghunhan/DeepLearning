# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        #self.W = np.random.randn(2,3) # 정규분포로 초기화
        self.W=np.array([[0.47355232, 0.9977393, 0.84668094], [0.85557411, 0.03563661, 0.69422093]])
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
print(net.predict(x))
print(net.loss(x,t))

f = lambda w: net.loss(x, t)

lr=0.01
decay_lamda=0.1

for i in range(100000):
    dW = numerical_gradient(f, net.W)
    dW=dW+decay_lamda*net.W
    net.W=net.W-lr*dW;

print(net.W)
print(net.predict(x))
print(net.loss(x,t))