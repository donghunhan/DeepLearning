# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient_2d


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

x = np.array([[0.6, 0.9], [0.3, 0.1], [0.2, 0.8]]) # Example for mini batch 
t = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

net = simpleNet()
p=net.predict(x)
#print(p)

f = lambda w: net.loss(x, t)
dW = numerical_gradient_2d(f, net.W)

print(net.loss(x,t))  
# 0.9782901931921137  <- W11 : 0.47365232 x+h  이 weight에 대한 x 입력 3개에 대한 평균손실함수값
# 0.9782773340893526  <- W11 : 0.47345232 x-h

#W11 편미분 기울기
print((0.9782901931921137-0.9782773340893526)/(0.0001*2))   # 0.06429551380582321

print(dW)

#[[ 0.06429551 -0.00300119 -0.06129432]
# [-0.04087501  0.11065906 -0.06978405]]