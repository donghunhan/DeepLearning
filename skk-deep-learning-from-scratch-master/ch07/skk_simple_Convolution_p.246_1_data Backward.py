import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.util import im2col, col2im

x=np.random.rand(1,3,7,7) # 1개의 3channel 7X7 data
print(x)
print("###################################################")
col=im2col(x,5,5,stride=1,pad=0)
print(col)

w=np.random.randint(4, size=75) #임의수 0-3 까지 75개 생성
W=w.reshape(1,3,5,-1) #Filters  W=w.reshape(1,3,5,5)
print(W)
col_W=W.reshape(1,-1).T
print(col_W)
out=np.dot(col,col_W)
print(out)
out=out.reshape(1,3,3,-1).transpose(0,3,1,2) #(1,1,3,3)
print(out)

print("----------------- Backward -- common\layers.py, line: 232 ----------------")
dout=out
FN, C, FH, FW = W.shape  #Filters  W.shape  (1,3,5,5)
dout = dout.transpose(0,2,3,1).reshape(-1, FN)
print(dout) # 9행 1열 line 18의 out 과 유사. 배치가 약간 다름
print("dout shape : ", dout.shape)
#db = np.sum(dout, axis=0)
#print(db)
dW = np.dot(col.T, dout) # 75X9 * 9X1 = 75X1 <-- col_W .T (line 16)
print(dW)
print("reshpe 전: ", dW.shape)
dW = dW.transpose(1, 0).reshape(FN, C, FH, FW) #원래의 Filter형상으로. Filter 학습을 위한 미분값
print("reshpe 후: ", dW.shape)

dcol = np.dot(dout, col_W.T)  # 9X1 * 1X75 = 9X75
print("dcol shape: ", dcol.shape)
dx = col2im(dcol, x.shape, FH, FW, stride=1,pad=0)
print("dx shape: ",dx.shape) # 미분이 적용된 입력이미지의 형상으로 만들어 Backward