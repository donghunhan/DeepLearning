import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.util import im2col

N=2
x=np.random.rand(N,3,8,8) # 1개의 3channel 7X7 data
print(x)
print("###################################################")

out_h=
out_w=
col=im2col(x,5,5,stride=1,pad=0)
print(col)

FN=3
w=np.random.randint(4, size=25*3*FN) #임의수 0-3 까지 75개 생성
W=w.reshape(FN,3,5,5) #Filters  W=w.reshape(1,3,5,5)
print(W)
col_W=W.reshape(FN,-1).T  # 교재 p.246 class Convolution  line 13 
print(col_W)
out=np.dot(col,col_W)
print(out)
out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2) #(2,3,4,4)
print(out)
