import numpy as np
# import sklearn
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.interpolate import Rbf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import my_tf_pkg as mtf

fig = plt.figure()
ax = fig.gca(projection='3d')

X,Y,Z = mtf.generate_meshgrid_h_add()
#Axes3D.plot_trisurf(X, Y, Z)
ax.plot_surface(X, Y, Z)

plt.show()
