def process_argv(argv):
    print 'print argv =',argv
    print 'len(argv) =',len(argv)
    experiment_name = 'tmp_experiment'
    if is_it_tensorboard_run(argv):
        if len(argv) == 6:
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True --logdir=/tmp/mdl_logs
            prefix = 'tmp_om'
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
            print 1
        else:
            # python main_nn.py --logdir=/tmp/mdl_logs
            prefix = 'tmp'
            slurm_jobid = 'TB'
            slurm_array_task_id = 'TB'
            job_number = 'TB'
            mdl_save = True
            print 2
    else:
        mdl_save = True
        if len(argv) == 7:
            # python main_nn.py      slurm_jobid     slurm_array_task_id     job_number      True      prefix      experiment_name
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name
            #
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
            prefix = argv[5]
            experiment_name = argv[6]
            print 3
        elif len(argv) == 6:
            # python main_nn.py      slurm_jobid     slurm_array_task_id     job_number      True      prefix
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
            prefix = argv[5]
            print 4
        elif len(argv) == 5:
            # python main_nn.py      slurm_jobid     slurm_array_task_id     job_number      True
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True
            prefix = 'tmp_om'
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
            print 5
        elif len(argv) == 4:
            # python main_nn.py slurm_jobid slurm_array_task_id job_number
            prefix = 'om'
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = True
            print 6
        elif len(argv) == 2: # if job_number
            # python main_nn.py job_number
            prefix='tmp'
            slurm_jobid = '0'
            slurm_array_task_id = '00'
            job_number = argv[1]
            print 7
        elif len(argv) == 1:
            # python main_nn.py
            prefix='tmp'
            slurm_jobid = '0'
            slurm_array_task_id = '00'
            job_number = 'test'
            print 8
        else:
            raise ValueError('Need to specify the correct number of params')
    return (prefix,slurm_jobid,slurm_array_task_id,job_number,mdl_save,experiment_name)

def is_it_tensorboard_run(argv):
    check_args = []
    for sys_arg in argv:
        check_args.extend(sys_arg.split('='))
    print check_args
    return '--logdir' in check_args
