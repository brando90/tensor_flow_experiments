import numpy as np
# import sklearn
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.interpolate import Rbf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import my_tf_pkg as mtf

fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y,Z = mtf.generate_meshgrid_h_add()

X_data, Y_data = mtf.make_mesh_grid_to_data_set(X,Y,Z)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.show()
