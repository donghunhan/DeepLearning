import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import *
from common.util import im2col, col2im


pool_h=2
pool_w=2
stride=2
pad=0
x_t=np.random.randint(6, size=2*(64+64+64))
x=x_t.reshape(2,3,8,8) 
#print(x)

N,C,H,W=x.shape
out_h=int(1+(H-pool_h)/stride)
out_w=int(1+(W-pool_w)/stride)
#print('out_h=', out_h,'out_w=', out_w )
col=im2col(x,pool_h,pool_w,stride,pad)
#print(col)
col=col.reshape(-1,pool_h*pool_w)
print(col)
arg_max = np.argmax(col, axis=1) 
print(arg_max)
print(arg_max.size)
print(arg_max.flatten())
out=np.max(col,axis=1)
print(out.size)
out=out.reshape(N, out_h,out_w,C).transpose(0,3,1,2)

print("-------------- Backward ---------------")
dout_t=np.random.randint(6, size=2*(16+16+16))
dout=dout_t.reshape(2,3,4,4) 

# Convolution Layer로부터 Backward 값을 전달받는다
# input image 형상 그대로 
print(dout.shape)

dout = dout.transpose(0, 2, 3, 1)
print(dout)
pool_size = pool_h * pool_w
print('dout.size=', dout.size, 'pool_size=', pool_size)
dmax = np.zeros((dout.size, pool_size))
print('dmax shape',dmax.shape)
#print('dmax[np.arange(arg_max.size), arg_max.flatten()]', dmax[np.arange(arg_max.size), arg_max.flatten()])
#print(dout.flatten())
dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
dx = col2im(dcol, x.shape, pool_h, pool_w, stride, pad)
print(dx)
print(dx.shape)
  

