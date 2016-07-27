def process_argv(argv):
    print 'print argv =',argv
    print 'len(argv) =',len(argv)
    experiment_name = 'tmp_experiment'
    train_S_type = 'multiple_S'
    units_list = [24,24]
    # units_list = [96,96]
    # task_name = 'qianli_func'
    # task_name = 'hrushikesh'
    # re_train = None
    # task_name = 'f_2D_task2'
    task_name = 'f_2d_task2_xsinglog1_x_depth2'
    # task_name = 'f_2d_task2_xsinglog1_x_depth3'
    # task_name = 'MNIST_flat'
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
        if len(argv) == 10:
            # python main_nn.py      slurm_jobid     slurm_array_task_id     job_number      True      prefix      experiment_name 3,3,3  multiple_S/single_S f_2D_task2
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name 3 multiple_S/single_S f_2D_task2
            #
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name 2,2 multiple_S f_2D_task2
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name 2,2 single_S f_2D_task2
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
            prefix = argv[5]
            experiment_name = argv[6]
            units =  argv[7].split(',')
            units_list = [ int(a) for a in units ]
            train_S_type = argv[8] # multiple_S/single_S
            task_name = argv[9]
            print 2.8
        elif len(argv) == 9:
            # python main_nn.py      slurm_jobid     slurm_array_task_id     job_number      True      prefix      experiment_name 3,3,3  multiple_S/single_S
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name 3 multiple_S/single_S
            #
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name 2,2 multiple_S
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name 2,2 single_S
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
            prefix = argv[5]
            experiment_name = argv[6]
            units =  argv[7].split(',')
            units_list = [ int(a) for a in units ]
            train_S_type = argv[8] # multiple_S/single_S
            print 2.8
        elif len(argv) == 8:
            # python main_nn.py      slurm_jobid     slurm_array_task_id     job_number      True      prefix      experiment_name 3,3,3
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True prefix experiment_name 3
            #
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
            prefix = argv[5]
            experiment_name = argv[6]
            units =  argv[7].split(',')
            units_list = [ int(a) for a in units ]
            print 2.9
        elif len(argv) == 7:
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
    return (prefix,slurm_jobid,slurm_array_task_id,job_number,mdl_save,experiment_name,units_list,train_S_type,task_name)

def is_it_tensorboard_run(argv):
    check_args = []
    for sys_arg in argv:
        check_args.extend(sys_arg.split('='))
    print check_args
    return '--logdir' in check_args
