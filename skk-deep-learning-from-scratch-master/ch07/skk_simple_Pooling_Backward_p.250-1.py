import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import *
from common.util import im2col, col2im


pool_h=2
pool_w=2
stride=2
pad=0
#x_t=np.random.randint(6, size=2*(16+16+16))
#x=x_t.reshape(2,3,4,4) 
#print(x)
x=np.array(
  [[[[3,3,4,2], [2,4,1,1], [5,4,0,5], [1,5,4,3]], 
  [[2,5,1,2], [4,0,1,4], [1,0,3,1], [0,0,2,3]],
  [[1,5,3,0], [5,1,3,4], [2,1,5,1], [3,5,4,4]]], 
  
  [[[4,2,3,2],[2,4,0,3], [3,0,1,4], [3,5,5,0]],    \
  [[4,2,1,3], [5,0,0,0], [4,3,2,5], [1,1,5,2]],    \
  [[5,0,5,5], [0,4,1,0], [1,2,5,5], [3,5,1,5]]]] )
print(x)

N,C,H,W=x.shape
out_h=int(1+(H-pool_h)/stride)
out_w=int(1+(W-pool_w)/stride)
#print('out_h=', out_h,'out_w=', out_w )
col=im2col(x,pool_h,pool_w,stride,pad)
print(col)
col=col.reshape(-1,pool_h*pool_w)
print(col)
arg_max = np.argmax(col, axis=1) 
print(arg_max)
print(arg_max.size)
#print(arg_max.flatten())
out=np.max(col,axis=1)
print(out)
out=out.reshape(N, out_h,out_w,C).transpose(0,3,1,2)
print(out)

print("-------------- Backward ---------------")
#dout_t=np.random.randint(6, size=2*(4+4+4))
#dout=dout_t.reshape(2,3,2,2) 
dout=np.array( [[[[3,4],[3,0]], [[4,3],[4,1]], [[3,2],[0,2]]],
                [[[3,5],[0,1]], [[2,0],[3,0]], [[0,5],[0,4]]]])
# Convolution Layer로부터 Backward 값을 전달받는다
# forward 형상 그대로 backward시 전달받는다
print(dout)

dout = dout.transpose(0, 2, 3, 1)
print(dout)
pool_size = pool_h * pool_w
print('dout.size=', dout.size, 'pool_size=', pool_size)
dmax = np.zeros((dout.size, pool_size))
print('dmax shape : ',dmax.shape)
print('dout.flatten() : ',dout.flatten() )
print('arg_max.flatten() : ',arg_max.flatten() )
dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
print('reshape전 :', dmax)
print('dout.shape :', dout.shape)
print('new dmax.shape :', dout.shape + (pool_size,))
dmax = dmax.reshape(dout.shape + (pool_size,)) 
print('reshape후 :', dmax)
print('new dmax.shape :', dmax.shape)
dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
print(dcol)
dx = col2im(dcol, x.shape, pool_h, pool_w, stride, pad)
print(dx)
print(dx.shape)
  

