import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.util import im2col

N=2  # data 개수
x=np.random.rand(2,3,8,8) # 2개의 3channel 8X8 data
#print(x)
print("###################################################")
col=im2col(x,5,5,stride=1,pad=0) # p.246 소스 코드 내
#print(col)

FN=3 # 필터 개수
w=np.random.randint(4, size=75*FN) #필터한개당 임의수 0-3 까지 75개 생성, 합 3개
W=w.reshape(3,3,5,5) #Filters  W=w.reshape(1,3,5,5)
#print(W)

#col_W=W.reshape(FN,-1)
#print('col_W :', col_W.shape)
#print(col_W)

col_W=W.reshape(FN,-1).T # p.246 소스 코드 내
print('col_W :', col_W.shape)
print(col_W)

out=np.dot(col,col_W) # p.246 소스 코드 내
print('reshape 전 out 형상 :', out.shape)
print(out)
out=out.reshape(N,4,4,-1).transpose(0,3,1,2) #(2,3,4,4) p.246 소스 코드 내
print(out)
print(out.shape)
