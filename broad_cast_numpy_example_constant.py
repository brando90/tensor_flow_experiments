import numpy as np

M = 5
D = 2
D1 = 3
x = np.random.rand(M,D)
W = np.random.rand(D,D1)
WW = np.sum( W, axis=0, keepdims=True) # D1 x 1
XX = np.sum( x, axis=1, keepdims=True) # D x 1

print WW
print XX

#P = WW.reshape(1,D1)+XX.reshape(5,1)
P = WW+XX
print P
