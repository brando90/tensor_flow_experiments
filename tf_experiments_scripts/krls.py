import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

import my_tf_pkg as mtf

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

def evalaute_models(data, stddevs, nb_centers_list, replace=False):
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
    N_train = X_train.shape[0]

    C_hats = []
    best_stddevs = []
    train_errors = [] # tp report train error of mdl
    cv_errors = [] # to choose best model
    test_errors = [] # to report true error of mdl
    Y_preds_trains = [] # for reconstructions
    Y_preds_cvs = [] # for reconstructions
    Y_preds_tests = [] # for reconstructions
    centers_list = [] # centers tried for init.
    for K in nb_centers_list:
        # get best std using CV
        mdl_best_params, errors, reconstructions = mtf.get_best_shape_and_mdl(K, data, stddevs, nb_inits=2)
        (C_hat, centers, best_stddev) = mdl_params
        (train_error, cv_error, test_error) = errors
        (Y_pred_train, Y_pred_cv, Y_pred_test) = reconstructions

        C_hats.append(C_hat)
        centers_list.append(centers)
        best_stddevs.append(best_stddev)
        train_errors.append(train_error)
        cv_errors.append(cv_error)
        test_errors.append(test_error)
        Y_preds_trains.append(Y_pred_train)
        Y_preds_cvs.append(Y_pred_cv)
        Y_preds_tests.append(Y_pred_test)
    # packing
    mdl_best_params = (C_hats, centers_list, best_stddevs)
    errors = (train_error, cv_error, test_error)
    reconstructions = (Y_pred_train, Y_pred_cv, Y_pred_test)
    return mdl_best_params, errors, reconstructions

def main():
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
    data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

    replace = False # with or without replacement
    stddevs = np.linspace(start=0.1, stop=4, num=50)
    nb_centers_list = [3, 6, 9, 12, 16, 24, 30, 39, 48, 55]
    centers_to_reconstruct_index = [1, 3, 5, 7, 9]
    colours = ['g','r','c','m','y']

    mdl_best_params, errors, reconstructions = evalaute_models(data, stddevs, nb_centers_list, replace=False)
    (C_hats, centers, best_stddevs) = mdl_params
    (train_error, cv_error, test_error) = errors
    (Y_pred_train, Y_pred_cv, Y_pred_test) = reconstructions

    # get things to reconstruct
    nb_centers_reconstruct = [nb_centers_list[i] for i in centers_to_reconstruct_index]
    rbf_predictions_reconstruct_train = [ Y_pred_train[i] for i in centers_to_reconstruct_index]
    rbf_predictions_reconstruct_test = [Y_pred_test[i] for i in centers_to_reconstruct_index]

    # plot reconstructions
    plot_reconstruction(fig_num=1, X_original=X_train,Y_original=Y_train, nb_centers=nb_centers_reconstruct, \
    rbf_predictions=rbf_predictions_reconstruct_train, colours=colours, markersize=3,title_name='Reconstruction_train')
    plot_reconstruction(fig_num=2, X_original=X_test,Y_original=Y_test, nb_centers=nb_centers_reconstruct, \
    rbf_predictions=rbf_predictions_reconstruct_test, colours=colours, markersize=3,title_name='Reconstruction_test')

    # plot errors
    plt.figure(3)
    plot_errors(nb_centers, rbf_errors_train,label='train_Errors', markersize=3,colour='b')
    plot_errors(nb_centers, rbf_errors_test,label='test_Errors', markersize=3,colour='r')

    results = {'mdl_params':mdl_params, 'errors':errors, 'reconstructions':reconstructions}
    path = './'
    json_file = 'krls_json'
    with open(path+json_file, 'w+') as f_json:
        json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
    print '\a' #makes beep

##

if __name__ == '__main__':
    main()
