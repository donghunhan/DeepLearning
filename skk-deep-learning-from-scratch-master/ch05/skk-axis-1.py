# coding: utf-8
import numpy as np
dY = np.array([[1,2,3],[4,5,6]])
# dB=np.sum(dY,axis=0)   # [5 7 9]
dB=np.sum(dY,axis=1)
print(dB) # [ 6 15]