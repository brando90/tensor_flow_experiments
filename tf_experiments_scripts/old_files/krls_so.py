import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

import my_tf_pkg as mtf

## Data sets
def get_kernel_matrix(x,W,S):
    beta = get_beta_np(S)
    Z = -beta*euclidean_distances(X=x,Y=W,squared=True)
    K = np.exp(Z)
    return K

# def get_index():
#     index = []
#     for i, center in enumerate(nb_centers):
#         target_center
#         if target_center == center:
#

N = 60000
low_x =-2*np.pi
high_x=2*np.pi
X = low_x + (high_x - low_x) * np.random.rand(N,1)
X_test = low_x + (high_x - low_x) * np.random.rand(N,1)
# f(x) = 2*(2(cos(x)^2 - 1)^2 -1
f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
Y = f(X)
Y_test = f(X_test)

#stddev = 1
stddev = 1.8
replace = False # with or without replacement
nb_centers_reconstruct = [6, 12, 24, 36, 48] # number of centers for RBF
nb_centers = [3, 6, 9, 12, 16, 24, 30, 39, 48, 55]
nb_centers = range(2,25)
colours = ['g','r','c','m','y']
#
rbf_predictions_reconstruct_train = []
rbf_predictions_reconstruct_test = []

#rbf_predictions_test = []
rbf_errors_test = []

#rbf_predictions_train = []
rbf_errors_train = []
for K in nb_centers:
    indices=np.random.choice(a=N,size=K,replace=replace) # choose numbers from 0 to D^(1)
    subsampled_data_points=X[indices,:] # M_sub x D

    beta = np.power(1.0/stddev,2)
    Kern = np.exp(-beta*euclidean_distances(X=X,Y=subsampled_data_points,squared=True)) # N_train x D^1
    (C,_,_,_) = np.linalg.lstsq(Kern,Y)

    #indices=np.random.choice(a=N,size=K,replace=replace) # choose numbers from 0 to D^(1)
    #subsampled_data_points=X_test[indices,:] # M_sub x D
    Kern_test = np.exp(-beta*euclidean_distances(X=X_test,Y=subsampled_data_points,squared=True)) # N_test x D^1

    Y_pred = np.dot( Kern , C )
    Y_pred_test = np.dot( Kern_test , C )

    #rbf_predictions_train.append(Y_pred)
    train_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
    rbf_errors_train.append(train_error)
    #rbf_predictions_test.append(Y_pred_test)
    test_error = sklearn.metrics.mean_squared_error(Y_test, Y_pred_test)
    rbf_errors_test.append(test_error)
    if K in nb_centers_reconstruct:
        rbf_predictions_reconstruct_train.append(Y_pred)
        rbf_predictions_reconstruct_test.append(Y_pred_test)

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

plot_reconstruction(fig_num=1, X_original=X,Y_original=Y, nb_centers=nb_centers_reconstruct, rbf_predictions=rbf_predictions_reconstruct_train, colours=colours, markersize=3,title_name='Reconstruction_train')
plot_reconstruction(fig_num=2, X_original=X_test,Y_original=Y_test, nb_centers=nb_centers_reconstruct, \
    rbf_predictions=rbf_predictions_reconstruct_test, colours=colours, markersize=3,title_name='Reconstruction_test')
plt.figure(3)
plot_errors(nb_centers, rbf_errors_train,label='train_Errors', markersize=3,colour='b')
plot_errors(nb_centers, rbf_errors_test,label='test_Errors', markersize=3,colour='r')
#
plt.legend()
plt.show()
