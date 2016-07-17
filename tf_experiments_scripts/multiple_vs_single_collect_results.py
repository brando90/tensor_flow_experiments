import json
import os
import pdb

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
    with open(path_to_json_file, 'r') as data_file:
        results = json.load(data_file)
    return results

def get_best_results_from_experiment(experiment_dirpath, list_runs_filenames):
    '''
        Returns the result structure for the current experiment from all the runs.

        list_runs_filenames = filenames list with potential runs
    '''
    best_cv_errors = float('inf')
    best_test_filname = None
    results_best_mdl = None
    for potential_run_filename in list_runs_filenames:
        # if current run=filenmae is a json struct then it has the results
        if 'json' in potential_run_filename:
            run_filename = potential_run_filename
            results_current_mdl = get_results(experiment_dirpath, run_filename)
            train_error, cv_error, test_error = get_errors_from(results)
            if cv_error < best_cv_errors:
                best_cv_errors = cv_error
                best_test_filname = run_filename
                results_best_mdl = results_current_mdl
    return results_best_mdl_result, best_test_filname

def get_results_for_experiments(path_to_experiments):
    experiment_results = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}
    for (experiment_dir, _, potential_runs) in os.walk(path_to_experiments):
        # if current dirpath is a valid experiment and not . (itself)
        if (experiment_dir != path_to_experiments):
            best_results, best_results_filename = get_best_results_from_experiment(experiment_dirpath=experiment_dir,list_runs_filename=potential_runs)
            nb_units = results_best_mdl['dims'][1]
            experiment_results[nb_units] = {'best_results':best_results,'best_results_filename':best_results_filename}
    return experiment_results
##
experiment = '/HBF1_multiple'
path_to_experiments = './om_results_test_experiments/'+experiment
multiple_experiment_results = get_results_for_experiments(path_to_experiments)

experiment = '/HBF1_single'
path_to_experiments = './om_results_test_experiments/'+experiment
single_experiment_results = get_results_for_experiments(path_to_experiments)

path_to_experiments = path_root + folder_experiments
print 'path_to_experiments ', path_to_experiments
# for current experiment, find which is the best run model (best cv)

#for each experiment

single = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}

print 'multiple ', multiple
#print 'best_test_filname ', best_test_filname
