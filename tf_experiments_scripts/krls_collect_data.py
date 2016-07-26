import sys
import pdb
import datetime

import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import json

import my_tf_pkg as mtf
#import my_tf_pkg.plotting_1D as plt1d

def main(argv):
        (_, task_name, prefix, experiment_name, nb_inits, nb_rbf_shapes, units)  = argv
        nb_inits, nb_rbf_shapes = int(nb_inits), int(nb_rbf_shapes)
        units_list =  units.split(',')
        nb_centers_list = [ int(a) for a in units_list ]
        date = datetime.date.today().strftime("%B %d").replace (" ", "_")
        print 'task_name ', task_name

        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
        data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
        print X_train.shape

        replace = False # with or without replacement
        stddevs = np.linspace(start=0.01, stop=10, num=nb_rbf_shapes)
        print 'number of RBF stddev tried:', len(stddevs)
        print 'start stddevs: ', stddevs

        mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean = mtf.evalaute_models(data, stddevs, nb_centers_list, replace=False, nb_inits=nb_inits)
        (C_hat_bests, centers_bests, best_stddevs) = mdl_best_params
        print 'best_stddevs: ',best_stddevs
        (train_errors_bests, cv_errors_bests, test_errors_bests) = errors_best
        (train_errors_means,cv_errors_means,test_errors_means, train_error_stds,cv_error_stds,test_error_stds) = errors_stats
        (train_errors_bests, cv_errors_bests, test_errors_bests) = [ list(err_list) for err_list in errors_best]
        (train_errors_means,cv_errors_means,test_errors_means, train_error_stds,cv_error_stds,test_error_stds) = [ list(err_list) for err_list in errors_stats]
        #(Y_pred_train_best, _, Y_pred_test_best) = reconstructions_best
        print 'train_errors_means: ', train_errors_means

        dir_path = './%s_experiments/%s_%s'%(prefix,date,experiment_name)
        mtf.make_and_check_dir(dir_path)
        # save rbf
        rbf_params_loc = dir_path+'/rbf_params_%s_%s'%(date,experiment_name)
        np.savez(rbf_params_loc,C_hat_bests=C_hat_bests,centers_bests=centers_bests,best_stddevs=best_stddevs,units_list=np.array(nb_centers_list))
        # save errors
        result_loc = dir_path+'/results_json_%s_%s'%(date,experiment_name)
        results = {'nb_centers_list':nb_centers_list}
        mtf.load_results_dic(results,train_errors_bests=train_errors_bests,cv_errors_bests=cv_errors_bests,test_errors_bests=test_errors_bests, train_errors_means=train_errors_means,cv_errors_means=cv_errors_means,test_errors_means=test_errors_means, train_error_stds=train_error_stds,cv_error_stds=cv_error_stds,test_error_stds=test_error_stds)
        with open(result_loc, 'w+') as f_json:
            json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
        #mtf.save_workspace(filename=result_loc,names_of_spaces_to_save=dir(),dict_of_values_to_save=locals())

if __name__ == '__main__':
    # python krls_collect_data.py f_2D_task2 tmp_krls krls_experiment_name_test 2 2 2,3,4
    # python krls_collect_data.py f_2d_task2_xsinglog1_x_depth2 tmp_krls krls_f_2d_task2_xsinglog1_x_depth2 2 2 2,3,4
    # python krls_collect_data.py f_2d_task2_xsinglog1_x_depth3 tmp_krls krls_f_2d_task3_xsinglog1_x_depth2 2 2 2,3,4
    # python krls_collect_data.py hrushikesh tmp_krls krls_hrushikesh 2 2 2,3,4

    # task_name = f_2d_task2_xsinglog1_x_depth2
    # task_name = f_2d_task2_xsinglog1_x_depth3
    argv = sys.argv
    main(argv)
    print '\a' #makes beep
