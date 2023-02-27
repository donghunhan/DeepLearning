import numpy as np
x=np.array( [[1.0, -0.5], [-2.0,3.0]] )
print(x)
mask=(x<=0)
print(mask)
x[mask]=0
print(x)