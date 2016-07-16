import json
import os

def process_file(dirpath,filename):
    if 'json' in filename:
        path_to_json_file = dirpath+'/'+filename
        with open(path_to_json_file, 'r') as data_file:
            results = json.load(data_file) #'train_errors':[], 'cv_errors':[],'test_errors':[]
            (train_errors, cv_errors, test_errors) = (results['train_errors'], results['cv_errors'], results['test_errors'])
            #
            (train_error, cv_error, test_error) = (train_errors[-1], cv_errors[-1], test_errors[-1])
    return train_error, cv_error, test_error, results

##

# get path to exeriments
prefix = 'om_results'
path_root = './%s_test_experiments'%(prefix)
folder_experiments = 'HBF1_multiple_vs_single'

path_to_experiments = path_root + folder_experiments
#for each experiment
for (dirpath, dirnames, filenames) in os.walk(path_to_experiments):
    # dirpath = current directory we are exploring (experiment dir)
    # dirnames = directories inside of dirpath
    # filenames = filenames inside dirpath (json files with experiments)

    # for current experiment, find which is the best run model (best cv)
    multiple = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}
    single = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}
    #
    best_cv_errors = float("inf")
    best_test_filname = None
    results_best_mdl = results_current_mdl
    # if current dirpath is a valid experiment
    if 'multiple' in dirpath or 'single' in dirpath:
        # go through all experiment runs in dirpath
        for filename  in filenames:
            (train_error, cv_error, test_error, results_current_mdl) = process_fine(dirpath,filename)
            if cv_error < best_cv_errors:
                best_cv_errors = cv_error
                best_test_filname = filename
                results_best_mdl = results_current_mdl
        # now we have the best model (and hyperparams)
        nb_units = results_best_mdl['dims'][1]
        #collect results in proper directory
        if 'multiple' in dirpath:
            experiment_dir = multiple
        elif 'single' in dirpath:
            experiment_dir = single
        else:
            raise('Invalid experiment: ', dirpath)
        experiment_dir[nb_units] = {}
        experiment_dir[nb_units]['results_best_mdl'] = results_best_mdl
        experiment_dir[nb_units]['best_test_filname'] = best_test_filname
    else:
        pass

print 'best_test_errors ', best_test_errors
print 'best_test_filname ', best_test_filname
