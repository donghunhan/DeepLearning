import sys, os
sys.path.append(os.pardir)
import numpy as np
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params('deep_convnet_params.pkl')

train_accuracy = network.accuracy(x_train, t_train)
test_accuracy = network.accuracy(x_test, t_test)

print('model train accuracy:',train_accuracy)
print('model test accuracy:',test_accuracy)