import numpy as np
# import sklearn
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.interpolate import Rbf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import my_tf_pkg as mtf

## get mesh data
#X,Y,Z = mtf.generate_meshgrid_h_add()
#X,Y,Z = mtf.generate_meshgrid_h_gabor()
func = mtf.f2D_func_task2_2
#func = mtf.f2D_func_task2_3
nb_recursive_layers = 3
#X,Y,Z = mtf.generate_meshgrid_f2D_task2(N=120000,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
data_sets = [None]
for l in range(1,nb_recursive_layers+1):
    (X,Y,Z) = mtf.generate_meshgrid_f2D_task2_func(func=func,N=60000,start_val=-1,end_val=1, nb_recursive_layers=l)
    #(X,Y,Z) =  mesh
    mesh = (X,Y,Z)
    data_sets.append(mesh)

## get data
# for l in range(1, nb_recursive_layers+1):
#     (X,Y,Z) =  data_sets[l]
#     X_data, Y_data = mtf.make_mesh_grid_to_data_set(X, Y, Z)
#     # get rbf
#     K, stddev = (100, 1)
#     C, Kern, centers = mtf.get_RBF(X=X_data, K=K, stddev=stddev, Y=Y_data)
#     Y_pred = mtf.rbf_predict(X_data, C, centers, stddev)
#     _,_,Z_pred = mtf.make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_pred)
#     #
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     surf = ax.plot_surface(X, Y, Z_pred, cmap=cm.coolwarm)
#     plt.title('RBF prediction')

#plt.title('nb_recursive_layers f2D_task2, nb_recursive_layers=%s'%nb_recursive_layers)
for l in range(1,nb_recursive_layers+1):
    (X,Y,Z) =  data_sets[l]
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    #surf = ax.plot_surface(X, Y, Z, cmap=cm.BrBG)
    plt.title('Original function depth = %d'%l)

plt.show()
