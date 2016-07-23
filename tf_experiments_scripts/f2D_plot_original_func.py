import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

## get mesh data
start_val, end_val = -1,1
N = 100
x_range = np.random.uniform(low=start_val, high=end_val, size=N)
y_range = np.random.uniform(low=start_val, high=end_val, size=N)
#x_range = np.linspace(start_val, end_val, N)
#y_range = np.linspace(start_val, end_val, N)
(X,Y) = np.meshgrid(x_range, y_range)
Z = np.sin(2*np.pi*X) + 4*np.power(Y - 0.5, 2) # h_add

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.title('Original function')

plt.show()
