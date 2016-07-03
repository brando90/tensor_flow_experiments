def process_argv(argv):
    if is_it_tensorboard_run(sys_arg):
        # python main_nn.py --logdir=/tmp/log_file_name
        prefix = 'tmp'
        slurm_jobid = 'TB'
        slurm_array_task_id = 'TB'
        job_number = 'TB'
        mdl_save = True
    elif len(argv) == 5:
        # python main_nn.py slurm_jobid slurm_array_task_id job_number True --logdir=/tmp/log_file_name
        prefix = 'tmp_om'
        slurm_jobid = argv[1]
        slurm_array_task_id = argv[2]
        job_number = argv[3]
        mdl_save = bool(argv[4])
    else:
        mdl_save = False
        if len(argv) == 5:
            # python main_nn.py slurm_jobid slurm_array_task_id job_number True
            prefix = 'om'
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
            mdl_save = bool(argv[4])
        if len(argv) == 4:
            # python main_nn.py slurm_jobid slurm_array_task_id job_number
            prefix = 'om'
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_number = argv[3]
        elif len(argv) == 2: # if job_number
            # python main_nn.py job_number
            prefix='tmp'
            slurm_jobid = '0'
            slurm_array_task_id = '00'
            job_number = argv[1]
        elif len(argv) == 1:
            job_number = 'test'
        else:
            raise ValueError('Need to specify 3 paramerers or 2')
    return (prefix,slurm_jobid,slurm_array_task_id,job_number,mdl_save)

def is_it_tensorboard_run(sys_arg):
    check_args = [ sys_arg.split('=') for sys_arg in sys.argv]
    return '--logdir' in check_args
