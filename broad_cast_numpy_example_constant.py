import numpy as np

M = 5
D = 2
D1 = 3
x = np.random.rand(M,D)
W = np.random.rand(D,D1)
WW = np.sum( W, axis=0)
XX = np.sum( x, axis=1)

P = WW+XX
print P
