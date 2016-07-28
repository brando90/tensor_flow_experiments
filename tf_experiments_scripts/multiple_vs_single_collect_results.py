import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import re

import krls

##
def get_list_errors(experiment_results):
    # experiment_results : units->results
    list_units = []
    list_test_errors = []
    for nb_units, results in experiment_results.iteritems():
        #print 'nb_units ', nb_units
        train_error, cv_error, test_error = get_errors_from(results)
        list_units.append(nb_units)
        list_test_errors.append(test_error)
    # sort based on first list
    list_units, list_test_errors = zip(*sorted(zip(list_units, list_test_errors)))
    return list_units, list_test_errors

def get_list_errors2(experiment_results):
    # experiment_results : units->results
    list_units = []
    list_train_errors = []
    list_test_errors = []
    for nb_units, results in experiment_results.iteritems():
        #print 'nb_units ', nb_units
        train_error, cv_error, test_error = get_errors_from(results)
        #print '--nb_units', nb_units
        #print 'train_error, cv_error, test_error ', train_error, cv_error, test_error
        list_units.append(nb_units)
        list_test_errors.append(test_error)
        list_train_errors.append(train_error)
    # sort based on first list
    print len(list_train_errors)
    print len(list_test_errors)
    _, list_train_errors = zip(*sorted(zip(list_units, list_train_errors)))
    list_units, list_test_errors = zip(*sorted(zip(list_units, list_test_errors)))
    return list_units, list_train_errors, list_test_errors

##

def get_errors_from(results):
    # get error lists
    (train_errors, cv_errors, test_errors) = (results['train_errors'], results['cv_errors'], results['test_errors'])
    # get most recent error
    (train_error, cv_error, test_error) = (train_errors[-1], cv_errors[-1], test_errors[-1])
    return train_error, cv_error, test_error

def get_results(dirpath, filename):
    train_error, cv_error, test_error = (None, None, None)
    results = None
    path_to_json_file = dirpath+'/'+filename
    #print 'path_to_json_file', path_to_json_file
    with open(path_to_json_file, 'r') as data_file:
        results = json.load(data_file)
    return results

def get_best_results_from_experiment(experiment_dirpath, list_runs_filenames):
    '''
        Returns the best result structure for the current experiment from all the runs.

        list_runs_filenames = filenames list with potential runs
    '''
    best_cv_errors = float('inf')
    best_cv_filname = None
    results_best = None
    final_train_errors = []
    final_cv_errors = []
    final_test_errors = []
    for potential_run_filename in list_runs_filenames:
        # if current run=filenmae is a json struct then it has the results
        if 'json' in potential_run_filename:
            run_filename = potential_run_filename
            results_current_run = get_results(experiment_dirpath, run_filename)
            train_error, cv_error, test_error = get_errors_from(results_current_run)
            final_train_errors.append(train_error)
            final_cv_errors.append(cv_error)
            final_test_errors.append(test_error)
            if cv_error < best_cv_errors:
                best_cv_errors = cv_error
                best_cv_filname = run_filename
                results_best = results_current_run
    return results_best, best_cv_filname, final_train_errors, final_cv_errors, final_test_errors

def get_results_for_experiments(path_to_experiments, verbose=False, split_string='jHBF1_'):
    '''
        Returns a dictionary containing the best results for each experiment
    '''
    print path_to_experiments
    experiment_results = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}
    for (experiment_dir, _, potential_runs) in os.walk(path_to_experiments):
        # if current dirpath is a valid experiment and not . (itself)
        if (experiment_dir != path_to_experiments):
            results_best, best_filename, final_train_errors, final_cv_errors, final_test_errors = get_best_results_from_experiment(experiment_dirpath=experiment_dir,list_runs_filenames=potential_runs)
            nb_units = results_best['dims'][1]
            #(left, right) = experiment_dir.split('jHBF1_')
            (left, right) = re.split('_jHBF[\d]*_',experiment_dir)
            if verbose:
                print '--'
                print right[0]
                print 'experiment_dir ', experiment_dir
                print 'potential_runs ', len(potential_runs)
                print 'type(potential_runs)', type(potential_runs)
                print 'nb_units ', nb_units
                print 'best_filename ', best_filename
            experiment_results[nb_units] = results_best
            experiment_results[nb_units]['final_train_errors'] = final_train_errors
            experiment_results[nb_units]['final_cv_errors'] = final_cv_errors
            experiment_results[nb_units]['final_test_errors'] = final_test_errors
    return experiment_results

def get_error_stats(experiment_results):
    '''
        Inserts (mutates) the dictionary results with mean std of errors.
    '''
    mean_train_errors = []
    mean_cv_errors = []
    mean_test_errors = []
    #
    std_train_errors = []
    std_cv_errors = []
    std_test_errors = []
    for nb_units in experiment_results.iterkeys():
        final_train_errors = experiment_results[nb_units]['final_train_errors']
        final_cv_errors = experiment_results[nb_units]['final_cv_errors']
        final_test_errors = experiment_results[nb_units]['final_test_errors']
        #
        mean_train_error = np.mean(final_train_errors)
        mean_cv_error = np.mean(final_cv_errors)
        mean_test_error = np.mean(final_test_errors)
        # experiment_results[nb_units]['mean_train_error'] = mean_train_error
        # experiment_results[nb_units]['mean_cv_error'] = mean_cv_error
        # experiment_results[nb_units]['mean_test_error'] = mean_test_error
        mean_train_errors.append(mean_train_error)
        mean_cv_errors.append(mean_cv_error)
        mean_test_errors.append(mean_test_error)
        #
        std_train_error = np.std(final_train_errors)
        std_cv_error = np.std(final_cv_errors)
        std_test_error = np.std(final_test_errors)
        # experiment_results[nb_units]['std_train_error'] = std_train_error
        # experiment_results[nb_units]['std_cv_error'] = std_cv_error
        # experiment_results[nb_units]['std_test_error'] = std_test_error
        std_train_errors.append(std_train_error)
        std_cv_errors.append(std_cv_error)
        std_test_errors.append(std_test_error)
    return mean_train_errors, std_train_errors, mean_test_errors, std_test_errors

def display_results_multiple_vs_single():
    ##
    experiment = '/multiple_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    multiple_experiment_results = get_results_for_experiments(path_to_experiments)
    mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    experiment = '/single_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    single_experiment_results = get_results_for_experiments(path_to_experiments)
    mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = get_error_stats(single_experiment_results)
    print mean_test_errors_single
    print std_test_errors_single

    #
    list_units_multiple, list_test_errors_multiple = get_list_errors(experiment_results=multiple_experiment_results)
    list_units_single, list_test_errors_single = get_list_errors(experiment_results=single_experiment_results)
    #
    plt.figure(3)
    krls.plot_errors(list_units_multiple, list_test_errors_multiple,label='HBF1 Multiple Standard Deviations', markersize=3, colour='r')
    krls.plot_errors(list_units_single, list_test_errors_single,label='HBF1 Single Errors Standard Deviations', markersize=3, colour='b')

    krls.plot_errors_and_bars(list_units_multiple, mean_test_errors_multiple, std_test_errors_multiple, label='Multiple Errors', markersize=3, colour='b')
    krls.plot_errors_and_bars(list_units_single, mean_test_errors_single, std_test_errors_single, label='Single Errors', markersize=3, colour='r')
    #
    plt.legend()
    plt.show()

def display_results_HBF2():
    ##
    experiment = '/multiple_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    experiment = '/hbf2_multiple_S'
    path_to_experiments = './om_results_test_experiments'+experiment
    single_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = get_error_stats(single_experiment_results)
    print mean_test_errors_single
    print std_test_errors_single

    #
    list_units_multiple, list_test_errors_multiple = get_list_errors(experiment_results=multiple_experiment_results)
    list_units_single, list_test_errors_single = get_list_errors(experiment_results=single_experiment_results)
    #
    plt.figure(3)
    krls.plot_errors(list_units_multiple, list_test_errors_multiple,label='HBF1 Multiple Standard Deviations', markersize=3, colour='r')
    krls.plot_errors(2*np.array(list_units_single), list_test_errors_single,label='HBF2 Multiple Errors Standard Deviations', markersize=3, colour='b')

    #krls.plot_errors_and_bars(list_units_multiple, mean_test_errors_multiple, std_test_errors_multiple, label='Multiple Errors', markersize=3, colour='b')
    #krls.plot_errors_and_bars(list_units_single, mean_test_errors_single, std_test_errors_single, label='Single Errors', markersize=3, colour='r')
    #
    plt.legend()
    plt.show()

def display_results_HBF1_vs_HBF1():
    ##
    experiment = '/multiple_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf1_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    experiment = '/single_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf1_single_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    experiment = '/hbf2_multiple_S'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf2_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = get_error_stats(single_experiment_results)

    experiment = '/hbf2_single_S'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf2_single_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = get_error_stats(single_experiment_results)

    #
    hbf1_list_units_multiple, hbf1_list_test_errors_multiple = get_list_errors(experiment_results=hbf1_multiple_experiment_results)
    hbf1_list_units_single, hbf1_list_test_errors_single = get_list_errors(experiment_results=hbf1_single_experiment_results)

    hbf2_list_units_multiple, hbf2_list_test_errors_multiple = get_list_errors(experiment_results=hbf2_multiple_experiment_results)
    hbf2_list_units_single, hbf2_list_test_errors_single = get_list_errors(experiment_results=hbf2_single_experiment_results)
    #
    plt.figure(3)
    krls.plot_errors(hbf1_list_units_multiple, hbf1_list_test_errors_multiple,label='HBF1 Multiple Standard Deviations', markersize=3, colour='r')
    krls.plot_errors(hbf1_list_units_multiple, hbf1_list_test_errors_single,label='HBF1 Single Errors Standard Deviations', markersize=3, colour='m')

    print len(hbf2_list_test_errors_multiple)
    print len(hbf1_list_units_multiple)
    krls.plot_errors(2*np.array(hbf2_list_units_multiple), hbf2_list_test_errors_multiple,label='HBF2 Multiple Standard Deviations', markersize=3, colour='b')
    krls.plot_errors(2*np.array(hbf2_list_units_multiple), hbf2_list_test_errors_single,label='HBF2 Single Errors Standard Deviations', markersize=3, colour='c')

    #krls.plot_errors_and_bars(list_units_multiple, mean_test_errors_multiple, std_test_errors_multiple, label='Multiple Errors', markersize=3, colour='b')
    #krls.plot_errors_and_bars(list_units_single, mean_test_errors_single, std_test_errors_single, label='Single Errors', markersize=3, colour='r')
    #
    plt.legend()
    plt.show()

def display_results_HBF1_task2():
    ##
    # experiment = '/multiple_S_task2_HP_hbf2'
    # path_to_experiments = './om_results_test_experiments'+experiment
    # hbf1_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    path_to_experiments = './om_results_test_experiments/multiple_S_task2_HP_hbf2'
    hbf1_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    #print hbf1_multiple_experiment_results

    #
    list_units, list_train_errors, list_test_errors = get_list_errors2(experiment_results=hbf1_multiple_experiment_results)
    #hbf1_list_units_single, hbf1_list_test_errors_single = get_list_errors(experiment_results=hbf1_single_experiment_results)

    #
    plt.figure(3)
    print 'hbf1_list_units_multiple: ', list_units
    print 'list_train_errors: ', list_train_errors
    print 'list_test_errors: ', list_test_errors
    krls.plot_errors(list_units, list_train_errors,label='HBF1 not shared HBF shape', markersize=3, colour='b')
    krls.plot_errors(list_units, list_test_errors,label='HBF1 not shared HBF shape', markersize=3, colour='r')

    plt.legend()
    plt.show()

def display_results_HBF1_xsinglog1_x():
    path_to_experiments = './om_results_test_experiments/task_27_july_NN1_depth_2_1000'
    nn1_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)

    path_to_experiments = './om_results_test_experiments/task_27_july_NN2_depth_2_1000'
    nn2_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)

    path_to_experiments = './om_results_test_experiments/task_27_july_NN3_depth_2_1000'
    nn3_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)

    #
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = get_list_errors2(experiment_results=nn2_multiple_experiment_results)
    nn3_list_units, nn3_list_train_errors, nn3_list_test_errors = get_list_errors2(experiment_results=nn3_multiple_experiment_results)

    #
    plt.figure(3)
    #
    list_units = np.array(nn1_list_units)
    print list_units
    krls.plot_errors(list_units, nn1_list_train_errors,label='NN1 train', markersize=3, colour='b')
    krls.plot_errors(list_units, nn1_list_test_errors,label='NN1 test', markersize=3, colour='c')
    #
    list_units = 2*np.array(nn2_list_units)
    print list_units
    krls.plot_errors(list_units, nn2_list_train_errors,label='NN2 train', markersize=3, colour='r')
    krls.plot_errors(list_units, nn2_list_test_errors,label='NN2 test', markersize=3, colour='m')
    #
    list_units = 3*np.array(nn3_list_units)
    print list_units
    krls.plot_errors(list_units, nn3_list_train_errors,label='NN3 train', markersize=3, colour='g')
    krls.plot_errors(list_units, nn3_list_test_errors,label='NN3 test', markersize=3, colour='y')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    display_results_HBF1_xsinglog1_x()
