import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import operator

import my_tf_pkg as mtf

def get_best_shape_and_mdl(K, data, stddevs, nb_inits=1):
    '''
    get best shape (hypothesis class) and mdl/hypothesis for fixed number of centers.
    also report its train error and generalization (test) error
    '''
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data

    train_errors = [] # tp report train error of mdl
    cv_errors = [] # to choose best model
    test_errors = [] # to report true error of mdl
    Y_preds_trains = [] # for reconstructions
    Y_preds_cvs = [] # for reconstructions
    Y_preds_tests = [] # for reconstructions
    centers_tried = [] # centers tried for init.
    stddevs_list_for_runs = []
    for _,stddev  in enumerate(stddevs):
        for i in xrange(nb_inits):
            #get subsampled_data_points for centers of RBF
            centers, _ = get_subsampled_points(X_train,K,replace=False)
            centers_tried.append(centers)
            #form Kernel Matrix
            Kern_train = get_kernel_matrix(X_train,centers, stddev) # N_train x D^1
            Kern_cv = get_kernel_matrix(X_cv,centers,stddev)
            Kern_test = get_kernel_matrix(X_test,centers,stddev)
            # train RBF
            C_hat = get_krls_coeffs(Kern_train,Y_train)
            # evluate RBF
            Y_pred_train = np.dot(Kern_train,C_hat)
            Y_preds_trains.append(Y_pred_train)
            train_error = sklearn.metrics.mean_squared_error(Y_train, Y_pred_train)
            train_errors.append(train_error)

            Y_pred_cv = np.dot(Kern_cv,C_hat)
            Y_preds_cvs.append(Y_pred_cv)
            cv_error = sklearn.metrics.mean_squared_error(Y_cv, Y_pred_cv)
            cv_errors.append(cv_error)

            Y_pred_test = np.dot(Kern_test,C_hat)
            Y_preds_tests.append(Y_pred_test)
            test_error = sklearn.metrics.mean_squared_error(Y_test, Y_pred_test)
            test_errors.append(test_error)
            #
            stddevs_list_for_runs.append(stddev)
    # get mdl had lowest CV
    min_index, _ = get_min(cv_errors)
    # get statistics of mdl model with best CV
    train_error = train_errors[min_index]
    cv_error = cv_errors[min_index] #min_cv
    test_error = test_errors[min_index]
    # std error
    train_error_std = np.std(train_errors)
    cv_error_std = np.std(cv_errors)
    test_error_std = np.std(test_errors)
    # shape of gaussian
    best_stddev = stddevs_list_for_runs[min_index]
    # get reconstructions
    Y_pred_train = Y_preds_trains[min_index]
    Y_pred_cv = Y_preds_cvs[min_index]
    Y_pred_test = Y_preds_tests[min_index]
    # centers
    centers =centers_tried[min_index]
    # packing
    mdl_best_params = (C_hat, centers, best_stddev)
    errors = (train_error, cv_error, test_error, train_error_std, cv_error_std, test_error_std)
    reconstructions = (Y_pred_train, Y_pred_cv, Y_pred_test)
    return mdl_best_params, errors, reconstructions

def get_subsampled_points(X,K,replace=False):
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
    centers/subsampled_data_points = the centers that are used for forming kernel function (columns)
    stddev = width/shape/std of Gaussian/RBF
    '''
    beta = np.power(1.0/stddev,2) #precision
    Kern = np.exp(-beta*euclidean_distances(X=X,Y=centers,squared=True)) # N_train x D^1
    return Kern

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


# def get_best_shape_and_mdl(K, data, stddevs):
#     '''
#     get best shape (hypothesis class) and mdl/hypothesis for fixed number of centers.
#     also report its train error and generalization (test) error
#     '''
#     (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data
#
#     cv_errors = [] # to choose best model
#     train_errors = [] # tp report train error of mdl
#     test_errors = [] # to report true error of mdl
#     Y_preds_trains = [] # for reconstructions
#     Y_preds_cvs = [] # for reconstructions
#     Y_preds_tests = [] # for reconstructions
#     centers_tried = [] # centers tried for init.
#     for _,stddev  in enumerate(stddevs):
#         #get subsampled_data_points for centers of RBF
#         centers, _ = get_subsampled_points(X_train,K,replace=False)
#         centers_tried.append(centers)
#         #form Kernel Matrix
#         Kern_train = get_kernel_matrix(X_train,centers, stddev) # N_train x D^1
#         Kern_cv = get_kernel_matrix(X_cv,centers,stddev)
#         Kern_test = get_kernel_matrix(X_test,centers,stddev)
#         # train RBF
#         C_hat = get_krls_coeffs(Kern_train,Y_train)
#         # evluate RBF
#         Y_pred_train = np.dot(Kern_train,C_hat)
#         Y_preds_trains.append(Y_pred_train)
#         train_error = sklearn.metrics.mean_squared_error(Y_train, Y_pred_train)
#         train_errors.append(train_error)
#
#         Y_pred_cv = np.dot(Kern_cv,C_hat)
#         Y_preds_cvs.append(Y_pred_cv)
#         cv_error = sklearn.metrics.mean_squared_error(Y_cv, Y_pred_cv)
#         cv_errors.append(cv_error)
#
#         Y_pred_test = np.dot(Kern_test,C_hat)
#         Y_preds_tests.append(Y_pred_test)
#         test_error = sklearn.metrics.mean_squared_error(Y_test, Y_pred_test)
#         test_errors.append(test_error)
#     # get mdl had lowest CV
#     min_index, min_cv = get_min(cv_errors)
#     # get statistics of mdl model with best CV
#     train_error = train_errors(min_index)
#     cv_error = min_cv
#     test_error = test_errors(min_index)
#     best_stddev = stddev(min_index)
#     # get reconstructions
#     Y_pred_train = Y_preds_trains(min_index)
#     Y_pred_cv = Y_preds_cvs(min_index)
#     Y_pred_test = Y_preds_tests(min_index)
#     # centers
#     centers =centers_tried[min_index]
#     #
#     mdl_best_params = (C_hat, centers, best_stddev)
#     errors = (train_error, cv_error, test_error)
#     reconstructions = (Y_pred_train, Y_pred_cv, Y_pred_test)
#     return mdl_best_params, errors, reconstructions
