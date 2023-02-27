import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.util import im2col

pad = 0
stride = 1
FH = 5
FW = 5
H,W = 8,8 # input image
N=2 # input image 개수
x=np.random.rand(N,3,H,W) # N개의 3channel 8X8 data
print(x)
print("###################################################")
out_h = int(1+(H+2*pad-FH)/stride)
out_w = int(1+(W+2*pad-FW)/stride)
#out_h = 1+8-5
#out_w = 1+8-5
col=im2col(x,5,5,stride=1,pad=0)
print(col)

FN=3
w=np.random.randint(4, size=25*3*FN) #임의수 0-3 까지 75*FN 개 생성
W=w.reshape(FN,3,5,5) #Filters  W=w.reshape(FN,3,5,5)
print(W)
col_W=W.reshape(FN,-1).T
print(col_W)
out=np.dot(col,col_W)
print(out)
out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2) #(0,1,3,3)
print(out)

