import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import operator
#import time
import pdb
#pdb.set_trace()

import my_tf_pkg as mtf

def find_index_value(list_vals, target):
    for i,val in enumerate(list_vals):
        if val ==  target:
            return i,val
    return None

def find_closest_to_value(list_vals, target):
    #min(myList, key=lambda x:abs(x-myNumber))
    current_dist = abs(target - list_val[0])
    best_dist = current_dist
    current_index = 0
    best_index = 0
    best_val = list_val[0]
    for i,val in enumerate(list_vals):
        current_dist = abs(target - val)
        if current_dist < best_dist:
            best_dist = current_dist
            best_index = i
            best_val = val
    return best_index, best_val

def evalaute_models(data, stddevs, nb_centers_list, replace=False, nb_inits=1):
    print 'evalauting models, nb_inits %s '%(nb_inits)
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data
    N_train = X_train.shape[0]

    # errors of best models
    train_errors_bests = []
    cv_errors_bests = []
    test_errors_bests = []
    # stats
    train_errors_means = []
    cv_errors_means = []
    test_errors_means = []
    train_errors_stds = []
    cv_errors_stds = []
    test_errors_stds = []
    # models
    C_hat_bests = []
    centers_bests = []
    best_stddevs = []
    # models TODO
    C_hat_means = []
    centers_means = []
    mean_stddevs = []
    # reconstructions for each center
    Y_preds_trains_bests = [] # for reconstructions
    Y_preds_cvs_bests = [] # for reconstructions
    Y_preds_tests_bests = [] # for reconstructions
    # TODO
    Y_pred_train_means = []
    Y_pred_cv_means = []
    Y_pred_test_means = []
    Y_pred_train_stds = []
    Y_pred_cv_stds = []
    Y_pred_test_stds = []
    for K in nb_centers_list:
        print '----center ', K
        # get best std using CV
        mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean = mtf.get_best_shape_and_mdl(K, data, stddevs, nb_inits=nb_inits)
        (C_hat_best, centers_best, best_stddev) = mdl_best_params
        (C_hat_mean, centers_mean, mean_stddev) = mdl_mean_params
        (train_error_best, cv_error_best, test_error_best) = errors_best
        (train_error_mean, cv_error_mean, test_error_mean, train_error_std, cv_error_std, test_error_std) = errors_stats
        (Y_pred_train_best, Y_pred_cv_best, Y_pred_test_best) = reconstructions_best
        (Y_pred_train_mean, Y_pred_cv_mean, Y_pred_test_mean, Y_pred_train_std, Y_pred_cv_std, Y_pred_test_std) = reconstructions_mean
        # record best
        train_errors_bests.append(train_error_best)
        cv_errors_bests.append(cv_error_best)
        test_errors_bests.append(test_error_best)
        # record mean
        train_errors_means.append(train_error_mean)
        cv_errors_means.append(cv_error_mean)
        test_errors_means.append(test_error_mean)
        train_errors_stds.append(train_error_std)
        cv_errors_stds.append(cv_error_std)
        test_errors_stds.append(test_error_std)
        # record best models
        C_hat_bests.append(C_hat_best)
        centers_bests.append(centers_best)
        best_stddevs.append(best_stddev)
        # reconstructions
        Y_preds_trains_bests.append(Y_pred_train_best)
        Y_preds_cvs_bests.append(Y_pred_cv_best)
        Y_preds_tests_bests.append(Y_pred_test_best)
    # packing
    mdl_best_params = (C_hat_bests, centers_bests, best_stddevs)
    mdl_mean_params = (C_hat_means, centers_means, mean_stddevs) # TODO
    errors_best = (train_errors_bests, cv_errors_bests, test_errors_bests)
    errors_stats = (train_errors_means, cv_errors_means, test_errors_means, train_errors_stds, cv_errors_stds, test_errors_stds)
    reconstructions_best = (Y_preds_trains_bests, Y_preds_cvs_bests, Y_preds_tests_bests)
    reconstructions_mean = (Y_pred_train_means, Y_pred_cv_means, Y_pred_test_means, Y_pred_train_stds, Y_pred_cv_stds, Y_pred_test_stds) # TODO
    return mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean

def get_best_shape_and_mdl(K, data, stddevs, nb_inits=1):
    '''
    get best shape (hypothesis class) and mdl/hypothesis for fixed number of centers.
    also report its train error and generalization (test) error
    '''
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data

    train_errors = [] # tp report train error of mdl
    cv_errors = [] # to choose best model
    test_errors = [] # to report true error of mdl
    #
    Y_preds_trains = [] # for reconstructions
    Y_preds_cvs = [] # for reconstructions
    Y_preds_tests = [] # for reconstructions
    #
    C_hats = []
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
            C_hats.append(C_hat)
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
    #mean_index = find_closest_to_value(list_vals=, target=) TODO
    # best params
    C_hat_best = C_hats[min_index]
    centers_best = centers_tried[min_index]
    best_stddev = stddevs_list_for_runs[min_index]
    # mean params TODO
    C_hat_mean = None
    centers_mean = None
    mean_stddev = best_stddev
    # get errors of mdl model with best CV
    train_error_best = train_errors[min_index]
    cv_error_best = cv_errors[min_index] #min_cv
    test_error_best = test_errors[min_index]

    # stats error
    (i,_) = find_index_value(list_vals=stddevs_list_for_runs, target=best_stddev)
    train_errors = train_errors[i:i+nb_inits]
    cv_errors = cv_errors[i:i+nb_inits]
    test_errors = test_errors[i:i+nb_inits]

    # mean
    train_error_mean = np.mean(train_errors)
    cv_error_mean = np.mean(cv_errors)
    test_error_mean = np.mean(test_errors)
    # std
    train_error_std = np.std(train_errors)
    cv_error_std = np.std(cv_errors)
    test_error_std = np.std(test_errors)

    ## get reconstructions
    # best reconstructions
    Y_pred_train_best = Y_preds_trains[min_index]
    Y_pred_cv_best = Y_preds_cvs[min_index]
    Y_pred_test_best = Y_preds_tests[min_index]
    # mean reconstructions TODO
    Y_pred_train_mean = None
    Y_pred_cv_mean = None
    Y_pred_test_mean = None
    # std reconstructions
    Y_pred_train_std = None
    Y_pred_cv_std = None
    Y_pred_test_std = None
    # packing
    mdl_best_params = (C_hat_best, centers_best, best_stddev)
    mdl_mean_params = (C_hat_mean, centers_mean, mean_stddev)
    errors_best = (train_error_best, cv_error_best, test_error_best)
    errors_stats = (train_error_mean, cv_error_mean, test_error_mean, train_error_std, cv_error_std, test_error_std)
    reconstructions_best = (Y_pred_train_best, Y_pred_cv_best, Y_pred_test_best)
    reconstructions_mean = (Y_pred_train_mean, Y_pred_cv_mean, Y_pred_test_mean, Y_pred_train_std, Y_pred_cv_std, Y_pred_test_std)
    return mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean

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
    #subsampled_data_points,_,_ = mtf.get_kpp_init(X,K,random_state=None)
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

def get_RBF(X, K, stddev, Y):
    centers,_ = get_subsampled_points(X,K,replace=False)
    Kern = get_kernel_matrix(X,centers, stddev)
    C = get_krls_coeffs(Kern, Y)
    return C, Kern, centers

def rbf_predict(X_data, C, centers, stddev):
    Kern = get_kernel_matrix(X_data,centers, stddev) # N_train x D^1
    # evluate RBF
    Y_pred = np.dot(Kern,C)
    return Y_pred

##

def get_min(values):
    min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    return min_index, min_value

def get_max(values):
    max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
    return max_index, max_value
