import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.util import im2col, col2im

print("----------------- forward  -- common\layers.py, line: 214 ------------------")
x=np.random.rand(10,3,7,7) # 10개의 3channel 7X7 data
#print(x)

col=im2col(x,5,5,stride=1,pad=0)
#print(col)

w=np.random.randint(4, size=375)
W=w.reshape(5,3,5,-1) #Filters  W=w.reshape(5,3,5,5)
#print(W)
col_W=W.reshape(5,-1).T # 5X75
#print(col_W)
out=np.dot(col,col_W)
#print(out) # 90행 5열
out=out.reshape(10,3,3,-1).transpose(0,3,2,1) #(10,5,3,3)
#print(out)

print("----------------- Backward -- common\layers.py, line: 232 ----------------")
dout=out
FN, C, FH, FW = W.shape  #Filters  W.shape  (5,3,5,5)
dout = dout.transpose(0,2,3,1).reshape(-1, FN)
#print(dout) # 90행 5열 line 18의 out 과 유사. 배치가 약간 다름
db = np.sum(dout, axis=0)
#print(db)
dW = np.dot(col.T, dout) # 75X90 * 90X5 = 75X5 <-- col_W .T (line 16)
#print(dW)
dW = dW.transpose(1, 0).reshape(FN, C, FH, FW) #5X75로 바꾼후 원래의 Filter형상으로. Filter 학습을 위한 미분값
print(dW.shape)
dcol = np.dot(dout, col_W.T)  # 90X5 * 5X75 = 90X75
print(dcol.shape)
dx = col2im(dcol, x.shape, FH, FW, stride=1,pad=0)
print(dx.shape) # 미분이 적용된 입력이미지의 형상으로 만들어 Backward