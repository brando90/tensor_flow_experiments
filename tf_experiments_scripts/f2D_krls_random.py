import numpy as np
# import sklearn
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.interpolate import Rbf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import my_tf_pkg as mtf

## get mesh data
#X,Y,Z = mtf.generate_random_meshgrid_h_gabor()
#X,Y,Z = mtf.random_random_gen_h_add()
nb_recursive_layers = 2
X,Y,Z = mtf.generate_random_meshgrid_f2D_task2(N=120000,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
## get data
X_data, Y_data = mtf.make_mesh_grid_to_data_set(X, Y, Z)
## get rbf
K, stddev = (1000, 1)
C, Kern, centers = mtf.get_RBF(X=X_data, K=K, stddev=stddev, Y=Y_data)
Y_pred = mtf.rbf_predict(X_data, C, centers, stddev)
#
X_pred,Y_pred,Z_pred = mtf.make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_pred)
#
Xp,Yp,Zp = mtf.make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_data)
X_data, Y_data = mtf.make_mesh_grid_to_data_set(Xp,Yp,Zp)
Xp,Yp,Zp = mtf.make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_data)

#plt.title('nb_recursive_layers f2D_task2, nb_recursive_layers=%s'%nb_recursive_layers)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Xp, Yp, Zp, cmap=cm.coolwarm)
plt.title('Original function')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_pred, Y_pred, Z_pred, cmap=cm.coolwarm)
plt.title('RBF prediction')

plt.show()
