import numpy as np

class Relu: # Common/Layers.py
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

x = np.array([[0,1.0,-0.5],[-2.0,3.0,0]])

r = Relu()
f = r.forward(x)
print(f)

y = np.array([[2.0,-1.0,-0.4],[5.1,-1.0,0.3]])
b = r.backward(y)
print(b)