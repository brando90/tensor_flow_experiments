import sys
import pdb

import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
# matplotlib.pyplot as plt
import json

import my_tf_pkg as mtf
#import my_tf_pkg.plotting_1D as plt1d

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
        #(Y_pred_train_best, _, Y_pred_test_best) = reconstructions_best

        mtf.save_workspace(filename=result_loc,names_of_spaces_to_save=dir(),dict_of_values_to_save=locals())
        print '\a' #makes beep

if __name__ == '__main__':
    # python krls_collect_data.py f_2D_task2 ./tmp_result_krls_f_2D_task2_results nb_inits nb_rbf_shapes units_list
    # python krls_collect_data.py f_2D_task2 ./tmp_result_krls_f_2D_task2_results 30 30 2,4,6,8,12,14,16,18,20,22,24,26,28,30
    # python krls_collect_data.py f_2D_task2 ./tmp_result_krls_f_2D_task2_results 3 3 2,3,4
    argv = sys.argv
    main(argv)
    print '\a' #makes beep
