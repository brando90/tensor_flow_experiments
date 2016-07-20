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
#X,Y,Z = mtf.generate_meshgrid_h_add()
#X,Y,Z = mtf.generate_meshgrid_h_gabor()
nb_recursive_layers = 2
X,Y,Z = mtf.generate_meshgrid_f2D_task2(N=120000,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)

plt.title('nb_recursive_layers f2D_task2, nb_recursive_layers=%s'%nb_recursive_layers)
X_data, Y_data = mtf.make_mesh_grid_to_data_set(X,Y,Z)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.show()
