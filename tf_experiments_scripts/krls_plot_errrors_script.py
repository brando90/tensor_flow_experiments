import sys
import pdb
import json

import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
#import json

import my_tf_pkg as mtf

import my_tf_pkg.plotting_1D as plt1d
import matplotlib.pyplot as plt

def main(rbf_params_filename,errors_filename,task_name):
    #_, filename = argv
    print 'errors_filename: ', errors_filename
    with open(errors_filename) as data_file:
        results = json.load(data_file)
    #
    nb_centers_list = results['nb_centers_list']
    train_errors_bests = results['train_errors_bests']
    test_errors_bests = results['test_errors_bests']

    train_errors_means = results['train_errors_means']
    test_errors_means = results['test_errors_means']

    train_error_stds = results['train_error_stds']
    test_error_stds = results['test_error_stds']

    print 'nb_centers_list: ', nb_centers_list
    print 'train_errors_bests: ', train_errors_bests
    print 'test_errors_bests: ', test_errors_bests

    # best stddevs
    npzfile = np.load(rbf_params_filename)
    best_stddevs = npzfile['best_stddevs']
    print 'best_stddevs: ', best_stddevs

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

if __name__ == '__main__':
    # frameworkpython krls_plot_errrors_script.py

    #### TMP
    # experiments_root = './tmp_krls_experiments'
    # experiment_dir = '/July_24_krls_f_2d_task2_xsinglog1_x_depth2'
    # results_filename = '/results_json_July_24_krls_f_2d_task2_xsinglog1_x_depth2'
    # rbf_params_filename = '/rbf_params_July_24_krls_f_2d_task2_xsinglog1_x_depth2.npz'

    # experiments_root = './tmp_krls_experiments'
    # experiment_dir = '/July_24_krls_mnist_test'
    # results_filename = '/results_json_July_24_krls_mnist_test'
    # rbf_params_filename = '/rbf_params_July_24_krls_mnist_test.npz'

    #### OM
    task_name = 'MNIST_flat'
    experiments_root = './om_krls_experiments'
    experiment_dir = '/July_24_krls_MNIST_flat_50_50_units_6_12_24_48_96_182_246_360_std_search'
    results_filename = '/results_json_July_24_krls_MNIST_flat_50_50_units_6_12_24_48_96_182_246_360_std_search'
    rbf_params_filename = '/rbf_params_July_24_krls_MNIST_flat_50_50_units_6_12_24_48_96_182_246_360_std_search.npz'

    ## singlog1_x_depth_2
    # task_name = 'f_2d_task2_xsinglog1_x_depth2'
    # experiments_root = './om_krls_experiments'
    # experiment_dir = '/July_24_krls_task2_xsinglog1_x_depth_2_100_100_D_6_12_24_48_96_182_246_364'
    # results_filename = '/results_json_July_24_krls_task2_xsinglog1_x_depth_2_100_100_D_6_12_24_48_96_182_246_364'
    # rbf_params_filename = '/rbf_params_July_24_krls_task2_xsinglog1_x_depth_2_100_100_D_6_12_24_48_96_182_246_364.npz'

    ## singlog1_x_depth_3
    # task_name = 'f_2d_task2_xsinglog1_x_depth3'
    # experiments_root = './om_krls_experiments'
    # experiment_dir = '/July_24_krls_task2_xsinglog1_x_depth_3_50_50_D_6_12_24_48_96_182_246_364'
    # results_filename = '/results_json_July_24_krls_task2_xsinglog1_x_depth_3_50_50_D_6_12_24_48_96_182_246_364'
    # rbf_params_filename = '/rbf_params_July_24_krls_task2_xsinglog1_x_depth_3_50_50_D_6_12_24_48_96_182_246_364.npz'

    main(rbf_params_filename=experiments_root+experiment_dir+rbf_params_filename,errors_filename=experiments_root+experiment_dir+results_filename, task_name=task_name)
    #main(sys.argv)
    print '\a' #makes beep
