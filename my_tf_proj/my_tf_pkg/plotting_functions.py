# import numpy as np
# import sklearn
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

#import my_tf_pkg as mtf

def plot_reconstruction(fig_num, X_original,Y_original, nb_centers, rbf_predictions, colours, markersize=3, title_name='Reconstruction'):
    fig = plt.figure(fig_num)
    plt.xlabel('number of centers')
    plt.ylabel('Reconstruction')
    plt.title(title_name)
    plt.plot(X_original, Y_original,'bo', label='Original data', markersize=markersize)
    for i, Y_pred in enumerate(rbf_predictions):
        colour = colours[i]
        K = nb_centers[i]
        plt.plot(X_original, Y_pred, colour+'o', label='RBF'+str(K), markersize=markersize)

def plot_errors(nb_centers, rbf_errors,label='Errors', markersize=3, colour='b'):
    plt.xlabel('number of centers')
    plt.ylabel('squared error (l2 loss)')
    plt.plot(nb_centers, rbf_errors, colour, label=label, markersize=3)
    plt.plot(nb_centers, rbf_errors, colour+'o')

##

def plot_surface_example(X,Y,Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X,Y,Z = mtf.generate_meshgrid_h_add()
    #Axes3D.plot_trisurf(X, Y, Z)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.show()
