import json
import os
import pdb
import krls
import matplotlib.pyplot as plt

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
    for potential_run_filename in list_runs_filenames:
        # if current run=filenmae is a json struct then it has the results
        if 'json' in potential_run_filename:
            run_filename = potential_run_filename
            results_current_run = get_results(experiment_dirpath, run_filename)
            train_error, cv_error, test_error = get_errors_from(results_current_run)
            if cv_error < best_cv_errors:
                best_cv_errors = cv_error
                best_cv_filname = run_filename
                results_best = results_current_run
    return results_best, best_cv_filname

def get_results_for_experiments(path_to_experiments):
    '''
        Returns a dictionary containing the best results for each experiment
    '''
    experiment_results = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}
    for (experiment_dir, _, potential_runs) in os.walk(path_to_experiments):
        # if current dirpath is a valid experiment and not . (itself)
        if (experiment_dir != path_to_experiments):
            results_best, best_filename = get_best_results_from_experiment(experiment_dirpath=experiment_dir,list_runs_filenames=potential_runs)
            nb_units = results_best['dims'][1]
            (left, right) = experiment_dir.split('jHBF1_')
            print '--'
            #print experiment_dir.split('om_results_July_16_jHBF1_')
            print right[0]
            print 'experiment_dir ', experiment_dir
            print 'potential_runs ', len(potential_runs)
            print 'type(potential_runs)', type(potential_runs)
            #print 'potential_runs[0]', potential_runs[0]
            print 'nb_units ', nb_units
            print 'best_filename ', best_filename
            experiment_results[nb_units] = results_best
    return experiment_results

##
experiment = '/HBF1_multiple'
path_to_experiments = './om_results_test_experiments'+experiment
multiple_experiment_results = get_results_for_experiments(path_to_experiments)

#pdb.set_trace()

experiment = '/HBF1_single'
path_to_experiments = './om_results_test_experiments'+experiment
single_experiment_results = get_results_for_experiments(path_to_experiments)
#
list_units_multiple, list_test_errors_multiple = get_list_errors(experiment_results=multiple_experiment_results)
#print list_units_multiple, list_test_errors_multiple
list_units_single, list_test_errors_single = get_list_errors(experiment_results=single_experiment_results)
#print list_units_multiple, list_test_errors_single
#
plt.figure(3)
krls.plot_errors(list_units_multiple, list_test_errors_multiple,label='Errors', markersize=3, colour='b')
krls.plot_errors(list_units_single, list_test_errors_single,label='Errors', markersize=3, colour='b')
#
# plt.legend()
# plt.show()
