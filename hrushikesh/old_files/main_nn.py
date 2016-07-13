## run cmd to collect model: python main_nn.py --logdir=/tmp/mdl_logs
## show board on browser run cmd: tensorboard --logdir=/tmp/mdl_logs
## browser: http://localhost:6006/

import numpy as np
import tensorflow as tf
import shutil
import subprocess
import json
import sys
import datetime
import os
from sklearn.cross_validation import train_test_split

import my_tf_pkg as mtf
#from tensorflow.python import control_flow_ops
import time

#import namespaces as ns

#import winsound

def make_and_check_dir(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        # uncomment to make it raise an error when path is not a directory
        #if not os.path.isdir(path):
        #    raise
        pass

def load_results_dic(results,**kwargs):
    for key, value in kwargs.iteritems():
        results[key] = value
    return results

print sys.argv

results = {'train_errors':[], 'cv_errors':[],'test_errors':[]}
# slyurm values and ids
(prefix,slurm_jobid,slurm_array_task_id,job_number,mdl_save) = mtf.process_argv(sys.argv)
print 'prefix=%s,slurm_jobid=%s,slurm_array_task_id=%s,job_number=%s'%(prefix,slurm_jobid,slurm_array_task_id,job_number)
results['job_number'] = job_number
results['slurm_jobid'] = slurm_jobid
results['slurm_array_task_id'] = slurm_array_task_id
# randomness
tf_rand_seed = int(os.urandom(32).encode('hex'), 16)
tf.set_random_seed(tf_rand_seed)
results['tf_rand_seed'] = tf_rand_seed
## directory structure for collecting data for experiments
path_root = './%s_test_experiments'%(prefix)

date = datetime.date.today().strftime("%B %d").replace (" ", "_")
results['date'] = date

current_experiment_folder = '/%s_%s_j%s'%(prefix,date,job_number)
path = path_root+current_experiment_folder
#path = './%s_test_experiments/%s_%s_j%s'%(prefix,prefix,date,job_number)
## files & folders with data or mdls
errors_pretty = '/%s_errors_file_%s_slurm_sj%s.txt'%(prefix,date,slurm_array_task_id)
mdl_dir ='/%s_mdl_%s_slurm_sj%s'%(prefix,date,slurm_array_task_id)
json_file = '/%s_json_%s_slurm_sj%s'%(prefix,date,slurm_array_task_id)
tensorboard_data_dump_train = '/tmp/mdl_logs_H/train'
tensorboard_data_dump_test = '/tmp/mdl_logs_H/test'
print '==> tensorboard_data_dump_train: ', tensorboard_data_dump_train
print '==> tensorboard_data_dump_test: ', tensorboard_data_dump_test
print 'mdl_save',mdl_save
# try to make directory, if it exists do NOP
make_and_check_dir(path=path)
make_and_check_dir(path=path+mdl_dir)
make_and_check_dir(path=tensorboard_data_dump_train)
make_and_check_dir(path=tensorboard_data_dump_test)
# delete contents of tensorboard dir
shutil.rmtree(tensorboard_data_dump_train)
shutil.rmtree(tensorboard_data_dump_test)
# JSON results structure
results_dic = mtf.fill_results_dic_with_np_seed(np_rnd_seed=np.random.get_state(), results=results)

## Data sets
#(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
with open('patient_data_X_Y.json', 'r') as f_json:
    patients_data = json.load(f_json)

X = patients_data['1']['X']
Y = patients_data['1']['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
X_cv, Y_cv = X_test, Y_test
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = ( np.array(X_train), np.array(Y_train), np.array(X_cv), np.array(Y_cv), np.array(X_test), np.array(Y_test) )
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape
print '(N_train,D)', (N_train,D)
print '(N_test,D_out)', (N_test,D_out)

## HBF/NN params
#dims = [D,24,D_out]
dims = [D,3,3,D_out]
#dims = [D,4,4,4,D_out]
#dims = [D,24,24,24,24,D_out]
mu = len(dims)*[0.0]
std = len(dims)*[0.1]
#std = [None,1,1,1]
init_constant = 1.39
#b_init = len(dims)*[init_constant]
b_init = len(dims)*[init_constant]
#b_init = [None, 1, 1, None]
S_init = b_init
init_type = 'truncated_normal'
#init_type = 'data_init'
#init_type = 'kern_init'
#init_type = 'kpp_init'
#model = 'standard_nn'
model = 'hbf'
#
max_to_keep = 10

bn = False
if bn:
    phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON
else:
    phase_train = None

## Make Model
x = tf.placeholder(tf.float64, shape=[None, D], name='x-input') # M x D
nb_layers = len(dims)-1
nb_hidden_layers = nb_layers-1
print( '-----> Running model: %s. (nb_hidden_layers = %d, nb_layers = %d)' % (model,nb_hidden_layers,nb_layers) )
print( '-----> Units: %s)' % (dims) )
if model == 'standard_nn':
    #tensorboard_data_dump = '/tmp/standard_nn_logs'
    (inits_C,inits_W,inits_b) = mtf.get_initilizations_standard_NN(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("standardNN") as scope:
        mdl = mtf.build_standard_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
        mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
elif model == 'hbf':
    #tensorboard_data_dump = '/tmp/hbf_logs'
    (inits_C,inits_W,inits_S) = mtf.get_initilizations_HBF(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("HBF") as scope:
        mdl = mtf.build_HBF2(x,dims,(inits_C,inits_W,inits_S),phase_train)
        mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])

## Output and Loss
y = mdl
y_ = tf.placeholder(tf.float64, shape=[None, D_out]) # (M x D)
with tf.name_scope("L2_loss") as scope:
    l2_loss = tf.reduce_mean(tf.square(y_-y))

## train params
report_error_freq = 10
steps = 3000
M = 50 #batch-size
# steps = 30
# M = 30 #batch-size

optimization_alg = 'GD'
#optimization_alg = 'Momentum'
#optimization_alg = 'Adadelta'
#optimization_alg = 'Adam'
#optimization_alg = 'Adagrad'
#optimization_alg = 'RMSProp'
with tf.name_scope("train") as scope:
    starter_learning_rate = 0.9
    decay_rate = 0.9
    decay_steps = 10000
    staircase = True
    # decay_steps = 10000000
    # staircase = False
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

    # Passing global_step to minimize() will increment it at each step.
    if optimization_alg == 'GD':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = opt.minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Momentum':
        momentum = 0.9
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
        train_step = opt.minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adadelta':
        rho = 0.95
        opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=rho, epsilon=1e-08, name='Adadelta')
        train_step = opt.minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adam':
        beta1=0.9
        beta2=0.999
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08, name='Adam')
        train_step = opt.minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate)
        train_step = opt.minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'RMSProp':
        decay = 0.9
        momentum = 0.0
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=1e-10, name='RMSProp')
        train_step = opt.minimize(l2_loss, global_step=global_step)

with tf.name_scope("l2_loss") as scope:
  ls_scalar_summary = tf.scalar_summary("l2_loss", l2_loss)

def register_all_variables_and_grads(y):
    all_vars = tf.all_variables()
    grad_vars = opt.compute_gradients(y,all_vars) #[ (gradient,variable) ]
    for (dldw,v) in grad_vars:
        if dldw != None:
            prefix_name = 'derivative_'+v.name
            suffix_text = 'dJd'+v.name
            mtf.put_summaries(var=tf.square(dldw),prefix_name=prefix_name,suffix_text=suffix_text)

register_all_variables_and_grads(y)
## TRAIN
if phase_train is not None:
    #DO BN
    feed_dict_train = {x:X_train, y_:Y_train, phase_train: False}
    feed_dict_cv = {x:X_cv, y_:Y_cv, phase_train: False}
    feed_dict_test = {x:X_test, y_:Y_test, phase_train: False}
else:
    #Don't do BN
    feed_dict_train = {x:X_train, y_:Y_train}
    feed_dict_cv = {x:X_cv, y_:Y_cv}
    feed_dict_test = {x:X_test, y_:Y_test}

def get_batch_feed(X, Y, M, phase_train):
    mini_batch_indices = np.random.randint(M,size=M)
    Xminibatch =  X[mini_batch_indices,:] # ( M x D^(0) )
    Yminibatch = Y[mini_batch_indices,:] # ( M x D^(L) )
    if phase_train is not None:
        #DO BN
        feed_dict = {x: Xminibatch, y_: Yminibatch, phase_train: True}
    else:
        #Don't do BN
        feed_dict = {x: Xminibatch, y_: Yminibatch}
    return feed_dict

def print_messages(*args):
    for i, msg in enumerate(args):
        print ('-->msg %s: ', msg)

if tf.gfile.Exists('/tmp/mdl_logs'):
  tf.gfile.DeleteRecursively('/tmp/mdl_logs')
tf.gfile.MakeDirs('/tmp/mdl_logs')

# Add ops to save and restore all the variables.
if mdl_save:
    saver = tf.train.Saver(max_to_keep=max_to_keep)
start_time = time.time()
with open(path+errors_pretty, 'w+') as f_err_msgs:
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        #writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)
        train_writer = tf.train.SummaryWriter(tensorboard_data_dump_train, sess.graph)
        test_writer = tf.train.SummaryWriter(tensorboard_data_dump_test, sess.graph)

        sess.run( tf.initialize_all_variables() )
        for i in xrange(steps):
            ## Create fake data for y = W.x + b where W = 2, b = 0
            #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
            feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
            ## Train
            if i%report_error_freq == 0:
                (summary_str_train,train_error) = sess.run(fetches=[merged, l2_loss], feed_dict=feed_dict_train)
                cv_error = sess.run(fetches=l2_loss, feed_dict=feed_dict_cv)
                (summary_str_test,test_error) = sess.run(fetches=[merged, l2_loss], feed_dict=feed_dict_test)

                train_writer.add_summary(summary_str_train, i)
                test_writer.add_summary(summary_str_test, i)

                loss_msg = "Model *%s%s*, step %d/%d, training error %g, test error %g \n"%(model,nb_hidden_layers,i,steps,train_error,test_error)
                mdl_info_msg = "Opt: %s, BN %s, After %d/%d iteration, Init: %s \n" % (optimization_alg,bn,i,steps,init_type)
                print_messages(loss_msg, mdl_info_msg)
                # store results
                results['train_errors'].append(train_error)
                results['cv_errors'].append(cv_error)
                results['test_errors'].append(test_error)
                # write errors to pretty print
                f_err_msgs.write(loss_msg)
                f_err_msgs.write(mdl_info_msg)
                # save mdl
                if mdl_save:
                    save_path = saver.save(sess, path+mdl_dir+'/model.ckpt',global_step=i)
            sess.run(fetches=[merged,train_step], feed_dict=feed_dict_batch) #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
load_results_dic(results,git_hash=git_hash,dims=dims,mu=mu,std=std,init_constant=init_constant,b_init=b_init,S_init=S_init,\
    init_type=init_type,model=model,bn=bn,path=path,\
    tensorboard_data_dump_test=tensorboard_data_dump_test,tensorboard_data_dump_train=tensorboard_data_dump_train,\
    report_error_freq=report_error_freq,steps=steps,M=M,optimization_alg=optimization_alg,\
    starter_learning_rate=starter_learning_rate,decay_rate=decay_rate,staircase=staircase)

seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
## dump results to JSON
results['seconds'] = seconds
results['minutes'] = minutes
results['hours'] = hours
with open(path+json_file, 'w+') as f_json:
    json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
print '\a' #makes beep
