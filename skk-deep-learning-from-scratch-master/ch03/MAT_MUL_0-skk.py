# coding: utf-8

import numpy as np
A=np.array([[1,2,3,4],[4,5,6,7]])
B=np.array([1,2,3,4])
print(B.shape)
print(np.dot(A,B))
print(np.dot(A,B.T))

# (4,)
# [30 60]
# [30 60]