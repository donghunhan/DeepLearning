# import numpy as np
# print("-------------- 동일한 형상 데이터생성 -------------")
# #out=np.arange(48)  #0 부터
# out=np.arange(1,49) # start - (end-1)
# print(out)
# out=out.reshape(3,-1)
# print(out)
# out=out.T
# print("out.shpae:", out.shape)
# print(out)
# print("-------------------------------------------------")
# out=out.reshape(4,4,-1).transpose(2,0,1)
# #out=out.reshape(2,4,4,-1).transpose(0,3,1,2)
# print(out)


import numpy as np
print("-------------- 동일한 형상 데이터생성 -------------")
#out=np.arange(48)  #0 부터
out=np.arange(1,97) # start - (end-1)
print(out)
out=out.reshape(3,-1)
print(out)
out=out.T
print("out.shpae:", out.shape)
print(out)
print("-------------------------------------------------")
#out=out.reshape(4,4,-1).transpose(2,0,1)
out=out.reshape(2,4,4,-1).transpose(0,3,1,2)
print(out)