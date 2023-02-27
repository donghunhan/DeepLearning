import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.util import im2col


pool_h=2
pool_w=2
stride=2
pad=0

x_t=np.random.randint(6, size=2*(16+16+16))
x=x_t.reshape(2,3,4,4) 
print(x)

N,C,H,W=x.shape
out_h=int(1+(H-pool_h)/stride)
out_w=int(1+(W-pool_w)/stride)
print('out_h=', out_h,'out_w=', out_w )
col=im2col(x,pool_h,pool_w,stride,pad)
print(col)
col=col.reshape(-1,pool_h*pool_w)
print(col)
out=np.max(col,axis=1)
print(out)
out=out.reshape(N, out_h,out_w,C).transpose(0,3,1,2)
print(out)
