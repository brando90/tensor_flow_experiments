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

def main(filename,task_name):
    #_, filename = argv
    with open(filename) as data_file:
        results = json.load(data_file)
    #
    nb_centers_list = results['nb_centers_list']
    train_errors_bests = results['train_errors_bests']
    test_errors_bests = results['test_errors_bests']

    train_errors_means = results['train_errors_means']
    test_errors_means = results['test_errors_means']

    train_error_stds = results['train_error_stds']
    test_error_stds = results['test_error_stds']

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
    task_name = 'f_2D_task2'

    experiments_root = './om_krls_experiments'
    experiment_dir = '/July_21_krls_experiment_name_test'
    results_filename = '/results_json_July_21_krls_experiment_name_test'

    main(experiments_root+experiment_dir+results_filename,task_name)
    #main(sys.argv)
    print '\a' #makes beep
