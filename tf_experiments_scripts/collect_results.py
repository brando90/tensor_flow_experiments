import json
import os


prefix = 'om_results'
path_root = './%s_test_experiments'%(prefix)
date = 'July_16' # <- date
#job_name = 'HBF1_12_multiple_S' # <- job_name
job_name = 'HBF1_12_multiple_S_random_HP'
current_experiment_folder = '/%s_%s_j%s'%(prefix,date,job_name)
path = path_root+current_experiment_folder
#
best_test_errors = float("inf")
best_test_filname = None
#for slurm_array_task_id in range(1,101):
for filename in os.listdir(path):
    #mdl_dir ='/mdls_%s_%s_slurm_sj%s'%(prefix,date,slurm_array_task_id)
    #json_file = '/%s_json_%s_slurm_sj%s'%(prefix,date,slurm_array_task_id)
    #path_to_json_file = path + json_file
    if 'json' in filename:
        path_to_json_file = path+'/'+filename
        with open(path_to_json_file, 'r') as data_file:
            results = json.load(data_file)
            #'train_errors':[], 'cv_errors':[],'test_errors':[]
            train_errors = results['train_errors']
            cv_errors = results['cv_errors']
            test_errors = results['test_errors']
            test_error = test_errors[-1]
            print test_error
            if test_error < best_test_errors:
                best_test_errors = test_error
                best_test_filname = filename

print 'best_test_errors ', best_test_errors
print 'best_test_filname ', best_test_filname
