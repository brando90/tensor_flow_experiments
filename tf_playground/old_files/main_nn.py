## run cmd to collect model: python main_nn.py --logdir=/tmp/log_file_name
## show board on browser run cmd: tensorboard --logdir=/tmp/log_file_name
## browser: http://localhost:6006/

import numpy as np
import tensorflow as tf
import shutil
import subprocess
import json
import sys
import datetime
import os

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

#path = '/Users/brandomiranda/Documents/MATLAB/hbf_research/om_simulations/tensor_flow_experiments/tf_experiments_scripts/'
results = {'test_errors':[],'train_errors':[]}
if len(sys.argv) > 1:
    slurm_jobid = sys.argv[1]
    slurm_array_task_id = sys.argv[2]
    job_number = sys.argv[3]
else:
    slurm_jobid = '0'
    slurm_array_task_id = '0'
    job_number = sys.argv[1]
results['job_number'] = job_number
results['job_number'] = job_number
results['job_number'] = job_number
date = datetime.date.today().strftime("%B %d").replace (" ", "_")
path = './tmp_test_experiemtns/tmp_j%s'%(date,job_number)
#path = './om_experiments/'
make_and_check_dir(path)
#paths
errors_pretty = '/tmp_errors_file_%s_slurm_j%s.txt'%(date,slurm_array_task_id)
mdl_dir ='/tmp_mdl_%s_slurm_j%s'%(date,slurm_array_task_id)
make_and_check_dir(path=mdl_dir)
json_file = '/tmp_json_%s_slurm_j%s'%(date,slurm_array_task_id)
# JSON results structure
results_dic = mtf.fill_results_dic_with_np_seed(np_rnd_seed=np.random.get_state(), results=results)
results['date'] = date

## Data sets
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape

## NN params
#dims = [D,24,D_out]
dims = [D,24,24,D_out]
#dims = [D,24,24,24,D_out]
#dims = [D,24,24,24,24,D_out]
mu = len(dims)*[0.0]
std = len(dims)*[0.1]
init_constant = 1000
b_init = len(dims)*[init_constant]
S_init = b_init
init_type = 'truncated_normal'
#init_type = 'data_init'
#init_type = 'kern_init'
#init_args = ns.FrozenNamespace(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
#model = 'summed_nn'
#model = 'summed_hbf'
model = 'standard_nn'
model = 'hbf'

bn = True
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
    tensorboard_data_dump = '/tmp_standard_nn_logs'
    (inits_C,inits_W,inits_b) = mtf.get_initilizations_standard_NN(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("standardNN") as scope:
        mdl = mtf.build_standard_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
        mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
# elif model == 'summed_nn':
#     tensorboard_data_dump = '/tmp/summed_nn_logs'
#     (inits_C,inits_W,inits_b) = mtf.get_initilizations_summed_NN(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
#     with tf.name_scope("summNN") as scope:
#         mdl = mtf.build_summed_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
elif model == 'hbf':
    #tensorboard_data_dump = '/tmp/hbf_logs'
    tensorboard_data_dump = '/tmp_hbf_logs'
    (inits_C,inits_W,inits_S) = mtf.get_initilizations_HBF(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("HBF") as scope:
        mdl = mtf.build_HBF(x,dims,(inits_C,inits_W,inits_S),phase_train)
        mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
# elif model == 'summed_hbf':
#     tensorboard_data_dump = '/tmp/summed_hbf_logs'
#     (inits_C,inits_W,inits_S) = mtf.get_initilizations_summed_HBF(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
#     with tf.name_scope("summHBF") as scope:
#         mdl = mtf.build_summed_HBF(x,dims,(inits_C,inits_W,inits_S),phase_train)

tensorboard_dump_path = path+tensorboard_data_dump
make_and_check_dir(path= tensorboard_dump_path)
print '==> tensorboard_data_dump: ', tensorboard_dump_path
# delete contents of tensorboard dir
shutil.rmtree(tensorboard_dump_path)
## Output and Loss
y = mdl
y_ = tf.placeholder(tf.float64, shape=[None, D_out]) # (M x D)
with tf.name_scope("L2_loss") as scope:
    l2_loss = tf.reduce_mean(tf.square(y_-y))

## train params
report_error_freq = 10
steps = 50
M = 10 #batch-size
optimization_alg = 'GD'
optimization_alg = 'Momentum'
#optimization_alg = 'Adadelta'
#optimization_alg = 'Adam'
#optimization_alg = 'Adagrad'
#optimization_alg = 'RMSProp'
with tf.name_scope("train") as scope:
    starter_learning_rate = 0.001
    decay_rate = 0.9
    decay_steps = 1000
    staircase = True
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

def register_all_variables_and_grards(y):
    all_vars = tf.all_variables()
    for v in tf.all_variables():
        tf.histogram_summary('hist_'+v.name, v)
        if v.get_shape() == []:
            tf.scalar_summary('scal_'+v.name, v)

    grad_vars = opt.compute_gradients(y,all_vars) #[ (T(gradient),variable) ]
    for (dldw,v) in grad_vars:
        if dldw != None:
            tf.histogram_summary('hist_'+v.name+'dW', dldw)
            if v.get_shape() == [] or dldw.get_shape() == []:
                tf.scalar_summary('scal_'+v.name+'dW', dldw)
            l2norm_dldw = tf.reduce_mean(tf.square(dldw))
            tf.scalar_summary('scal_'+v.name+'dW_l2_norm', l2norm_dldw)

register_all_variables_and_grards(y)

## TRAIN
if phase_train is not None:
    #DO BN
    feed_dict_train = {x:X_train, y_:Y_train, phase_train: False}
    feed_dict_test = {x:X_test, y_:Y_test, phase_train: False}
else:
    #Don't do BN
    feed_dict_train = {x:X_train, y_:Y_train}
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

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=30)
start_time = time.time()
with open(path+errors_pretty, 'w+') as f_err_msgs:
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    f_err_msgs.write(git_hash)
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(path+mdl_dir, sess.graph)

        sess.run( tf.initialize_all_variables() )
        for i in xrange(steps):
            ## Create fake data for y = W.x + b where W = 2, b = 0
            #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
            feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
            ## Train
            if i%report_error_freq == 0:
                (summary_str_train,train_error) = sess.run(fetches=[merged, l2_loss], feed_dict=feed_dict_train)
                (summary_str_test,test_error) = sess.run(fetches=[merged, l2_loss], feed_dict=feed_dict_test)
                writer.add_summary(summary_str_train, i)

                loss_msg = "Model *%s%s*, step %d/%d, training error %g, test error %g \n"%(model,nb_hidden_layers,i,steps,train_error,test_error)
                mdl_info_msg = "Opt: %s, BN %s, After %d/%d iteration, Init: %s \n" % (optimization_alg,bn,i,steps,init_type)
                print_messages(loss_msg, mdl_info_msg)
                # store results
                results['train_errors'].append(train_error)
                results['test_errors'].append(test_error)
                # write errors to pretty print
                f_err_msgs.write(loss_msg)
                f_err_msgs.write(mdl_info_msg)
                # save mdl
                #save_path = saver.save(sess, path+'/tmp_mdls/model.ckpt',global_step=i)
                save_path = saver.save(sess, path+mdl_dir+'/model.ckpt',global_step=i)
            sess.run(fetches=[merged,train_step], feed_dict=feed_dict_batch)
            #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

load_results_dic(results,git_hash=git_hash,dims=dims,mu=mu,std=std,init_constant=init_constant,b_init=b_init,S_init=S_init,\
    init_type=init_type,model=model,bn=bn,path=path,tensorboard_data_dump=tensorboard_data_dump,\
    report_error_freq=report_error_freq,steps=steps,M=M,optimization_alg=optimization_alg,\
    starter_learning_rate=starter_learning_rate,decay_rate=decay_rate,staircase=staircase)

seconds = (time.time() - start_time)
minutes = seconds/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
## dump results to JSON
results['seconds'] = seconds
results['minutes'] = minutes
with open(path+json_file, 'w+') as f_json:
    json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
#winsound.Beep(Freq = 2500,Dur = 1000)
