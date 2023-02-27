import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.util import im2col

x=np.random.rand(10,3,7,7) # 10개의 3channel 7X7 data
#print(x)
print("###################################################")
col=im2col(x,5,5,stride=1,pad=0)
print(col.shape)

w=np.random.randint(4, size=375) # 0-3 임의수 발생
W=w.reshape(5,3,5,-1) #Filters  W=w.reshape(5,3,5,5)
print(W.shape)
col_W=W.reshape(5,-1).T
print(col_W.shape)
out=np.dot(col,col_W)
print(out.shape)
out=out.reshape(10,3,3,-1).transpose(0,3,1,2) #(10,5,3,3)
print(out.shape)
