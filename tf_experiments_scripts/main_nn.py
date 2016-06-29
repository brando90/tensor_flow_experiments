## run cmd to collect model: python main_nn.py --logdir=/tmp/log_file_name
## show board on browser run cmd: tensorboard --logdir=/tmp/log_file_name
## browser: http://localhost:6006/

import numpy as np
import tensorflow as tf

import my_tf_pkg as mtf
#from tensorflow.python import control_flow_ops
import time

import namespaces as ns

#import winsound

## Data sets
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape

## NN params
dims = [D,24,D_out]
#dims = [D,24,24,D_out]
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
model = 'standard_nn'
model = 'summed_nn'
model = 'hbf'
#model = 'summed_hbf'

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
    tensorboard_data_dump = '/tmp/standard_nn_logs'
    (inits_C,inits_W,inits_b) = mtf.get_initilizations_standard_NN(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("standardNN") as scope:
        mdl = mtf.build_standard_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
        mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
elif model == 'summed_nn':
    tensorboard_data_dump = '/tmp/summed_nn_logs'
    (inits_C,inits_W,inits_b) = mtf.get_initilizations_summed_NN(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("summNN") as scope:
        mdl = mtf.build_summed_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
elif model == 'hbf':
    tensorboard_data_dump = '/tmp/hbf_logs'
    (inits_C,inits_W,inits_S) = mtf.get_initilizations_HBF(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("HBF") as scope:
        mdl = mtf.build_HBF(x,dims,(inits_C,inits_W,inits_S),phase_train)
        mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
elif model == 'summed_hbf':
    tensorboard_data_dump = '/tmp/summed_hbf_logs'
    (inits_C,inits_W,inits_S) = mtf.get_initilizations_summed_HBF(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("summHBF") as scope:
        mdl = mtf.build_summed_HBF(x,dims,(inits_C,inits_W,inits_S),phase_train)

## Output and Loss
y = mdl
y_ = tf.placeholder(tf.float64, shape=[None, D_out]) # (M x D)
with tf.name_scope("L2_loss") as scope:
    l2_loss = tf.reduce_mean(tf.square(y_-y))

## train params
report_error_freq = 100
steps = 604000
M = 1000 #batch-size
optimization_alg = 'GD'
optimization_alg = 'Momentum'
#optimization_alg = 'Adadelta'
optimization_alg = 'Adam'
#optimization_alg = 'Adagrad'
#optimization_alg = 'RMSProp'
with tf.name_scope("train") as scope:
    starter_learning_rate = 0.00001
    decay_rate = 0.80
    decay_steps = 1000
    staircase = True
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

    # Passing global_step to minimize() will increment it at each step.
    if optimization_alg == 'GD':
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Momentum':
        momentum = 0.9
        train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adadelta':
        rho = 0.95
        train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=rho, epsilon=1e-08, name='Adadelta').minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adam':
        beta1=0.9
        beta2=0.999
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08, name='Adam').minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adagrad':
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'RMSProp':
        decay = 0.9
        momentum = 0.0
        train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=1e-10, name='RMSProp').minimize(l2_loss, global_step=global_step)

with tf.name_scope("l2_loss") as scope:
  ls_scalar_summary = tf.scalar_summary("l2_loss", l2_loss)

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

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=30)

start_time = time.time()
path_tf_exmperiments = '/Users/brandomiranda/Documents/MATLAB/hbf_research/om_simulations/tensor_flow_experiments/tf_experiments_scripts/'
with open(path_tf_exmperiments+'errors_file.txt', 'w+') as f:
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)

        sess.run( tf.initialize_all_variables() )
        for i in xrange(steps):
            ## Create fake data for y = W.x + b where W = 2, b = 0
            #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
            feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
            ## Train
            if i%report_error_freq == 0:
                train_result = sess.run([merged, l2_loss], feed_dict=feed_dict_train)
                summary_str_train = train_result[0]
                train_error = train_result[1]

                test_result = sess.run([merged, l2_loss], feed_dict=feed_dict_test)
                summary_str_test = test_result[0]
                test_error = test_result[1]

                writer.add_summary(summary_str_train, i)
                #print("step %d, training error %g"%(i, train_error))
                loss_msg = "Model *%s%s*, step %d, training error %g, test error %g \n"%(model, nb_hidden_layers, i, train_error,test_error)
                print loss_msg,
                mdl_info_msg = "Opt: %s, BN %s, After %d iteration, Init: %s \n" % (optimization_alg,bn,i,init_type)
                print mdl_info_msg,
                # write results
                f.write(loss_msg)
                f.write(mdl_info_msg)
                # save mdl
                save_path = saver.save(sess, path_tf_exmperiments+'/tmp_mdls/model.ckpt',global_step=i)
                #print("Model saved in file: %s" % save_path)
            sess.run(train_step, feed_dict=feed_dict_batch)
            #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


seconds = (time.time() - start_time)
minutes = seconds/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
#winsound.Beep(Freq = 2500,Dur = 1000)
