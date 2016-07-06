import numpy as np
import matplotlib.pyplot as plt

import my_tf_pkg as mtf

stddev = 2
beta = np.power(1.0/stddev,2)
t = 0
x = np.linspace(start=-5+t, stop=5-t, num=100)
gau = 10*np.exp(-beta*(x - t)**2) # N_train x D^1

plt.figure(1)
plt.plot(x, gau,'b-', label='Original data', markersize=3)
plt.legend()
plt.show()
