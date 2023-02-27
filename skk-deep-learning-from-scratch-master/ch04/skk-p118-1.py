import numpy as np
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    delta=1e-7
    return -np.sum(t*np.log(y+delta)) /batch_size

t=[[0,0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0]]
y=[[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0],[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]]
t1=[0,0,1,0,0,0,0,0,0,0]
y1=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))
print(cross_entropy_error(np.array(y1),np.array(t1)))