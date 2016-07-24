import sys
import pdb

import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import json

import my_tf_pkg as mtf
import my_tf_pkg.plotting_1D as plt1d

def main3(agv):
        (_, task_name, result_loc, nb_inits, nb_rbf_shapes, units)  = argv
        nb_inits, nb_rbf_shapes = int(nb_inits), int(nb_rbf_shapes)
        units_list =  units.split(',')
        nb_centers_list = [ int(a) for a in units_list ]

        print 'task_name ', task_name
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
        data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

        replace = False # with or without replacement
        stddevs = np.linspace(start=0.1, stop=3, num=nb_rbf_shapes)
        print 'number of RBF stddev tried:', len(stddevs)
        print 'start stddevs: ', stddevs

        mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean = mtf.evalaute_models(data, stddevs, nb_centers_list, replace=False, nb_inits=nb_inits)
        (C_hat_bests, centers_bests, best_stddevs) = mdl_best_params
        print 'best_stddevs: ',best_stddevs
        (train_errors_bests, _, test_errors_bests) = errors_best
        (train_errors_means,_,test_errors_means, train_error_stds,_,test_error_stds) = errors_stats
        #(Y_pred_train_best, _, Y_pred_test_best) = reconstructions_best

        #mtf.save_workspace(filename=result_loc,names_of_spaces_to_save=dir(),dict_of_values_to_save=locals())
        print '\a' #makes beep

def plot_reconstruction(fig_num, X_original,Y_original, nb_centers, rbf_predictions, colours, markersize=3, title_name='Reconstruction'):
    fig = plt.figure(fig_num)
    plt.xlabel('number of units')
    plt.ylabel('Reconstruction')
    plt.title(title_name)
    plt.plot(X_original, Y_original,'bo', label='Original data', markersize=markersize)
    for i, Y_pred in enumerate(rbf_predictions):
        colour = colours[i]
        K = nb_centers[i]
        plt.plot(X_original, Y_pred, colour+'o', label='RBF'+str(K), markersize=markersize)

def plot_one_func(fig_num, X_original,Y_original, markersize=3, title_name='Reconstruction'):
    fig = plt.figure(fig_num)
    plt.xlabel('number of units')
    plt.ylabel('Reconstruction')
    plt.title(title_name)
    plt.plot(X_original, Y_original,'bo', label='Original data', markersize=markersize)

def plot_errors(nb_centers, rbf_errors,label='Errors', markersize=3, colour='b'):
    plt.xlabel('number of units')
    plt.ylabel('squared error (l2 loss)')
    plt.plot(nb_centers, rbf_errors, colour, label=label, markersize=3)
    plt.plot(nb_centers, rbf_errors, colour+'o')
    plt.title("Erors vs centers")

def plot_errors_and_bars(nb_centers, rbf_errors, rbf_error_std, label='Errors', markersize=3, colour='b'):
    plt.xlabel('number of units')
    plt.ylabel('squared error (l2 loss)')
    plt.plot(nb_centers, rbf_errors, colour, label=label, markersize=3)
    plt.plot(nb_centers, rbf_errors, colour+'o')
    plt.errorbar(nb_centers, rbf_errors, yerr=rbf_error_std)
    plt.title("Erors vs units")

def main(argv):
    (_, task_name, result_loc, nb_inits, nb_rbf_shapes, units)  = argv
    nb_inits, nb_rbf_shapes = int(nb_inits), int(nb_rbf_shapes)
    units_list =  units.split(',')
    nb_centers_list = [ int(a) for a in units_list ]

    print 'task_name ', task_name
    #pdb.set_trace()
    #(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
    #task_name = 'qianli_func'
    #task_name = 'hrushikesh'
    #task_name = 'f_2D_task2'
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

    replace = False # with or without replacement
    #nb_rbf_shapes = 2 #<--
    stddevs = np.linspace(start=0.1, stop=3, num=nb_rbf_shapes)
    print 'number of RBF stddev tried:', len(stddevs)
    print 'start stddevs: ', stddevs
    #nb_centers_list = [3, 6, 9, 12, 16, 24, 30, 39, 48, 55]
    #nb_centers_list = [2, 4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #nb_centers_list = [2, 4]
    #nb_centers_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    #centers_to_reconstruct_index = [1, 3, 5, 7, 9]
    #centers_to_reconstruct_index = [1, 4, 7] # corresponds to centers 4, 12, 18
    #colours = ['g','r','c','m','y']

    #nb_inits = 2 #<--

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
    if task_name == 'qianli_func':
        plot_one_func(fig_num=1, X_original=X_train,Y_original=Y_train, markersize=3, title_name='Reconstruction')
        print 'plotting reconstruct for task %s'%task_name
        nb_centers_reconstruct = [nb_centers_list[i] for i in centers_to_reconstruct_index]
        rbf_predictions_reconstruct_train = [Y_pred_train_best[i] for i in centers_to_reconstruct_index]
        rbf_predictions_reconstruct_test = [Y_pred_test_best[i] for i in centers_to_reconstruct_index]
        colours = colours[0:len(centers_to_reconstruct_index)]
        #colours = [colours[i] for i in centers_to_reconstruct_index]
        # plot reconstructions
        print 'plotting reconstructions'
        plot_reconstruction(fig_num=1, X_original=X_train,Y_original=Y_train, nb_centers=nb_centers_reconstruct, \
        rbf_predictions=rbf_predictions_reconstruct_train, colours=colours, markersize=3,title_name='Reconstruction_train')
        plot_reconstruction(fig_num=2, X_original=X_test,Y_original=Y_test, nb_centers=nb_centers_reconstruct, \
        rbf_predictions=rbf_predictions_reconstruct_test, colours=colours, markersize=3,title_name='Reconstruction_test')
    elif task_name == 'f_2D_task2':
        print 'HERE'
        pass
        print 'plotting reconstruct for task %s'%task_name
    # plot show
    plt.legend()
    plt.show()

    #result_loc = './tmp_test_experiments/tmp_krls_workspace'
    mtf.save_workspace(filename=result_loc,names_of_spaces_to_save=dir(),dict_of_values_to_save=locals())
    print '\a' #makes beep

def main(argv):
    (_, task_name, result_loc, nb_inits, nb_rbf_shapes, units)  = argv
    nb_inits, nb_rbf_shapes = int(nb_inits), int(nb_rbf_shapes)
    units_list =  units.split(',')
    nb_centers_list = [ int(a) for a in units_list ]

    print 'task_name ', task_name
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

    replace = False # with or without replacement
    stddevs = np.linspace(start=0.1, stop=3, num=nb_rbf_shapes)
    print 'number of RBF stddev tried:', len(stddevs)
    print 'start stddevs: ', stddevs

    mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean = mtf.evalaute_models(data, stddevs, nb_centers_list, replace=False, nb_inits=nb_inits)
    (C_hat_bests, centers_bests, best_stddevs) = mdl_best_params
    print 'best_stddevs: ',best_stddevs
    (train_errors_bests, _, test_errors_bests) = errors_best
    (train_errors_means,_,test_errors_means, train_error_stds,_,test_error_stds) = errors_stats
    (Y_pred_train_best, _, Y_pred_test_best) = reconstructions_best

    # plot errors
    print 'plotting errors'
    plt1d.figure(n=1)
    plt1d.plot_errors(nb_centers_list, train_errors_bests,label='train_Errors_best', markersize=3,colour='b')
    plt1d.plot_errors(nb_centers_list, test_errors_bests,label='test_Errors_best', markersize=3,colour='r')
    plt1d.plot_errors_and_bars(nb_centers_list, train_errors_means, train_error_stds, label='train_Errors_average', markersize=3,colour='b')
    plt1d.plot_errors_and_bars(nb_centers_list, test_errors_means, test_error_stds, label='test_Errors_average', markersize=3,colour='r')

    # get things to reconstruct
    if task_name == 'qianli_func':
        print 'plotting reconstruct for task %s'%task_name
        pass
    elif task_name == 'f_2D_task2':
        print 'plotting reconstruct for task %s'%task_name
        print 'HERE'
        pass

    # plot show
    plt1d.show()

    #result_loc = './tmp_test_experiments/tmp_krls_workspace'
    mtf.save_workspace(filename=result_loc,names_of_spaces_to_save=dir(),dict_of_values_to_save=locals())
    print '\a' #makes beep

##

if __name__ == '__main__':
    # frameworkpython krls.py f_2D_task2 ./om_result_krls_f_2D_task2_results nb_inits nb_rbf_shapes units_list
    # frameworkpython krls.py f_2D_task2 ./om_result_krls_f_2D_task2_results 30 30 2,4,6,8,12,14,16,18,20,22,24,26,28,30
    # frameworkpython krls.py f_2D_task2 ./om_result_krls_f_2D_task2_results 2 2 2,4,6,8,12,14,16,18,20,22,24,26,28,30
    # frameworkpython krls.py f_2D_task2 ./om_result_krls_f_2D_task2_results 3 3 2,3,4

    # f_2d_task2_xsinglog1_x_depth2
    # task_name = f_2d_task2_xsinglog1_x_depth3
    argv = sys.argv
    main2(argv)
    print '\a' #makes beep
