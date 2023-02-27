# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # self.W = np.random.randn(2,3) # 정규분포로 초기화
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

#p=net.predict(x)
#t=np.array([0,0,1])
print(net.loss(x,t))  #f(x) 계산
#f = lambda w: net.loss(x, t)
#dW = numerical_gradient(f, net.W)
print((0.9280902109609267-0.9280463614466788)/(0.0001*2))



# 결과 ->  0.2192475712392561
#print(dW)
