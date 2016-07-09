import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import operator

import my_tf_pkg as mtf

def get_best_shape_and_mdl(K, data, stddevs):
    '''
    get best shape (hypothesis class) and mdl/hypothesis for fixed number of centers.
    also report its train error and generalization (test) error
    '''
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data

    cv_errors = [] # to choose best model
    train_errors = [] # tp report train error of mdl
    test_errors = [] # to report true error of mdl
    for _,stddev  in enumerate(stddevs):
        #get subsampled_data_points for centers of RBF
        centers, _ = get_sumsampled_points(X_train,K,replace=False)
        #form Kernel Matrix
        Kern_train = get_kernel_matrix(X_train,centers, stddev) # N_train x D^1
        Kern_cv = get_kernel_matrix(X_cv,centers,stddev)
        Kern_test = get_kernel_matrix(X_test,centers,stddev)
        # train RBF
        C_hat = get_krls_coeffs(Kern_train,Y_train)
        # evluate RBF
        Y_pred_train = np.dot(Kern_train,C_hat)
        train_error = sklearn.metrics.mean_squared_error(Y_train, Y_pred_train)
        train_errors.append(train_error)

        Y_pred_cv = np.dot(Kern_cv,C_hat)
        cv_error = sklearn.metrics.mean_squared_error(Y_cv, Y_pred)
        cv_errors.append(cv_error)

        Y_pred_test = np.dot(Kern_test,C_hat)
        test_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        test_errors.append(test_error)
    min_index, min_cv = get_min(cv_errors)
    train_error = train_errors(min_index)
    cv_error = min_cv
    test_error = test_errors(min_index)
    best_stddev = stddev(min_index)
    return C_hat, centers, best_stddev, train_error, cv_error, test_error

def get_sumsampled_points(X,K,replace=False):
    '''
    X = data set to randomly subsample
    K = number of centers.
    replace = subsampling with or without replacement
    '''
    #get subsampled_data_points
    N = X.shape[0] #N = range of indexes to choose from. Usually data set size.
    indices=np.random.choice(a=N,size=K,replace=replace) # choose numbers from 0 to D^(1)
    subsampled_data_points=X[indices,:] # M_sub x D
    return subsampled_data_points, indices

def get_kernel_matrix(X,centers, stddev):
    '''
    X = Data set to evaluate Kernel matrix (rows of Kern)
    subsampled_data_points = the centers that are used for forming kernel function (columns)
    stddev = width/shape/std of Gaussian/RBF
    '''
    beta = np.power(1.0/stddev,2) #precision
    Kern = np.exp(-beta*euclidean_distances(X=X,Y=subsampled_data_points,squared=True)) # N_train x D^1
    return

def get_krls_coeffs(Kern, Y):
    '''
    C = the learned coeffs (i.e. C is soln to KC = Y by least-squares)
    '''
    (C,_,_,_) = np.linalg.lstsq(Kern,Y)
    return C

def get_min(values):
    min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    return min_index, min_value

def get_max(values):
    max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
    return max_index, max_value
