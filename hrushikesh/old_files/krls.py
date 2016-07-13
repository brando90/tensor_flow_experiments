import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import json

import my_tf_pkg as mtf

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

def plot_one_func(fig_num, X_original,Y_original, markersize=3, title_name='Reconstruction'):
    fig = plt.figure(fig_num)
    plt.xlabel('number of centers')
    plt.ylabel('Reconstruction')
    plt.title(title_name)
    plt.plot(X_original, Y_original,'bo', label='Original data', markersize=markersize)

def plot_errors(nb_centers, rbf_errors,label='Errors', markersize=3, colour='b'):
    plt.xlabel('number of centers')
    plt.ylabel('squared error (l2 loss)')
    plt.plot(nb_centers, rbf_errors, colour, label=label, markersize=3)
    plt.plot(nb_centers, rbf_errors, colour+'o')
    plt.title("Erors vs centers")

def plot_errors_and_bars(nb_centers, rbf_errors, rbf_error_std, label='Errors', markersize=3, colour='b'):
    plt.xlabel('number of centers')
    plt.ylabel('squared error (l2 loss)')
    # plt.plot(nb_centers, rbf_errors, colour, label=label, markersize=3)
    # plt.plot(nb_centers, rbf_errors, colour+'o')
    plt.errorbar(nb_centers, rbf_errors, yerr=rbf_error_std)
    plt.title("Erors vs centers")

def main():
    #(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
    with open('patient_data_X_Y.json', 'r') as f_json:
        patients_data = json.load(f_json)

    X = patients_data['1']['X']
    Y = patients_data['1']['Y']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    X_cv, Y_cv = X_test, Y_test
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = ( np.array(X_train), np.array(Y_train), np.array(X_cv), np.array(Y_cv), np.array(X_test), np.array(Y_test) )
    (N_train,D) = X_train.shape
    (N_test,D_out) = Y_test.shape
    print '(N_train,D)', (N_train,D)
    print '(N_test,D_out)', (N_test,D_out)
    data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

    plot_one_func(fig_num=1, X_original=X_train,Y_original=Y_train, markersize=3, title_name='Reconstruction')

    replace = False # with or without replacement
    nb_rbf_shapes = 10
    stddevs = np.linspace(start=0.1, stop=3, num=nb_rbf_shapes)
    print 'start stddevs: ', stddevs
    #nb_centers_list = [3, 6, 9, 12, 16, 24, 30, 39, 48, 55]
    nb_centers_list = [2, 4, 6, 8, 12, 14, 16, 18, 20, 22]
    #centers_to_reconstruct_index = [1, 3, 5, 7, 9]
    centers_to_reconstruct_index = [1, 4, 7] # corresponds to centers 4, 12, 18
    colours = ['g','r','c','m','y']

    nb_inits = 10
    mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean = evalaute_models(data, stddevs, nb_centers_list, replace=False, nb_inits=nb_inits)
    (C_hat_bests, centers_bests, best_stddevs) = mdl_best_params
    print 'best_stddevs: ',best_stddevs
    (train_errors_bests, _, test_errors_bests) = errors_best
    (train_errors_means,_,test_errors_means, train_error_stds,_,test_error_stds) = errors_stats
    (Y_pred_train_best, _, Y_pred_test_best) = reconstructions_best

    # plot errors
    print 'plotting errors'
    plt.figure(3)
    plot_errors(nb_centers_list, train_errors_bests,label='train_Errors_best', markersize=3,colour='b')
    plot_errors(nb_centers_list, test_errors_bests,label='test_Errors_best', markersize=3,colour='r')
    plot_errors_and_bars(nb_centers_list, train_errors_means, train_error_stds, label='train_Errors_average', markersize=3,colour='b')
    plot_errors_and_bars(nb_centers_list, test_errors_means, test_error_stds, label='test_Errors_average', markersize=3,colour='r')

    # get things to reconstruct
    # print 'plotting reconstruct'
    # nb_centers_reconstruct = [nb_centers_list[i] for i in centers_to_reconstruct_index]
    # rbf_predictions_reconstruct_train = [Y_pred_train_best[i] for i in centers_to_reconstruct_index]
    # rbf_predictions_reconstruct_test = [Y_pred_test_best[i] for i in centers_to_reconstruct_index]
    # colours = colours[0:len(centers_to_reconstruct_index)]
    # #colours = [colours[i] for i in centers_to_reconstruct_index]
    # # plot reconstructions
    # print 'plotting reconstructions'
    # plot_reconstruction(fig_num=1, X_original=X_train,Y_original=Y_train, nb_centers=nb_centers_reconstruct, \
    # rbf_predictions=rbf_predictions_reconstruct_train, colours=colours, markersize=3,title_name='Reconstruction_train')
    # plot_reconstruction(fig_num=2, X_original=X_test,Y_original=Y_test, nb_centers=nb_centers_reconstruct, \
    # rbf_predictions=rbf_predictions_reconstruct_test, colours=colours, markersize=3,title_name='Reconstruction_test')
    #
    plt.legend()
    plt.show()

    # results = {'mdl_params':mdl_best_params, 'errors':errors, 'reconstructions':reconstructions}
    # path = './'
    # json_file = 'tmp_krls_json'
    # with open(path+json_file, 'w+') as f_json:
    #     json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
    # print '\a' #makes beep

##

if __name__ == '__main__':
    main()
