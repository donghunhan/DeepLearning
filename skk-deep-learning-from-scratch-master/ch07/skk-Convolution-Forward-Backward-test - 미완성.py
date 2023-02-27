import numpy as np
print("-------------- 동일한 형상 데이터생성 -------------")
#out=np.arange(48)  #0 부터
out=np.arange(1,49) # start - (end-1)
print(out)
out=out.reshape(3,-1)
print(out)
out=out.T
print("out.shpae:", out.shape)
print(out)
print("-------------------------------------------------")
#out=out.reshape(4,4,-1).transpose(2,0,1)
out=out.reshape(1,4,4,-1).transpose(0,3,1,2)
print(out)


print("----------------- Backward -- common\layers.py, line: 232 ----------------")
dout=out
#FN, C, FH, FW = W.shape  #Filters  W.shape  (1,3,5,5)
FN, C, FH, FW = (1,3,5,5)
dout = dout.transpose(0,2,3,1).reshape(-1, FN)
print(dout) # 9행 1열 line 18의 out 과 유사. 배치가 약간 다름
print("dout shape : ", dout.shape)
#db = np.sum(dout, axis=0)
#print(db)
#dW = np.dot(col.T, dout) # 75X9 * 9X1 = 75X1 <-- col_W .T (line 16)
""" print(dW)
print("reshpe 전: ", dW.shape)
dW = dW.transpose(1, 0).reshape(FN, C, FH, FW) #원래의 Filter형상으로. Filter 학습을 위한 미분값
print("reshpe 후: ", dW.shape)

dcol = np.dot(dout, col_W.T)  # 9X1 * 1X75 = 9X75
print("dcol shape: ", dcol.shape)
dx = col2im(dcol, x.shape, FH, FW, stride=1,pad=0)
print(dx.shape) # 미분이 적용된 입력이미지의 형상으로 만들어 Backward """