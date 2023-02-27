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
#print(x)

N,C,H,W=x.shape
out_h=int(1+(H-pool_h)/stride)
out_w=int(1+(W-pool_w)/stride)
print('out_h=', out_h,'out_w=', out_w )
col=im2col(x,pool_h,pool_w,stride,pad)
#print(col)
col=col.reshape(-1,pool_h*pool_w)
#print(col)
out=np.max(col,axis=1)
print(out)
out=out.reshape(N, out_h,out_w,C).transpose(0,3,1,2)
print(out)
print('Pooling layer output : ',out.shape) # batch size N=2

Poolingout=out.copy()
print('------------Affine Layer--------------')
params={}
params['W1']=0.01*np.random.randn(3*2*2, 5)
W1=params['W1']
print('Affine Layer - W1 shape : ',W1.shape)

Poolingout=Poolingout.reshape(Poolingout.shape[0],-1) 
#batch 입력을 전제로 한다. 입력 data 각각을 일차원으로 만들어 버린다. (N, -1) - common\layer.py, line:54
print('Reshaped Pooling Layer Output : ', Poolingout.shape)
Affineout=np.dot(Poolingout,W1)
print('Affine Layer Output : ', Affineout.shape)
print(Affineout)