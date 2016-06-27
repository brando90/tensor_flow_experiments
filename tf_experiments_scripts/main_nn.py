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

def get_initilizations_standard_NN(init_args):
    if init_args.init_type == 'truncated_normal':
        inits_W = [None]
        inits_b = [None]
        nb_hidden_layers = len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[init_args.dims[l-1],init_args.dims[l]], mean=init_args.mu[l], stddev=init_args.std[l], dtype=tf.float64) )
            inits_b.append( tf.constant(init_args.b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        l = len(init_args.dims)-1
        inits_C = [ tf.truncated_normal(shape=[init_args.dims[l-1],init_args.dims[l]], mean=init_args.mu, stddev=init_args.std, dtype=tf.float64) ]
    elif init_args.init_type  == 'data_init':
        X_train = init_args.X_train
        pass
    return (inits_C,inits_W,inits_b)

def get_initilizations_summed_NN(init_args):
    if init_args.init_type == 'truncated_normal':
        inits_W = [None]
        inits_b = [None]
        inits_C = [None]
        nb_hidden_layers = len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[1,init_args.dims[l]], mean=init_args.mu[l], stddev=init_args.std[l], dtype=tf.float64) )
            inits_b.append( tf.constant(init_args.b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            inits_C.append( tf.truncated_normal(shape=[init_args.dims[l],1], mean=init_args.mu[l], stddev=init_args.std[l], dtype=tf.float64) )
    elif init_args.init_type  == 'data_init':
        X_train = init_args.X_train
        pass
    return (inits_C,inits_W,inits_b)

def get_initilizations_summed_HBF(init_args):
    if init_args.init_type == 'truncated_normal':
        inits_W = [None]
        inits_S = [None]
        inits_C = [None]
        nb_hidden_layers = len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[1,init_args.dims[l]], mean=init_args.mu[l], stddev=init_args.std[l], dtype=tf.float64) )
            inits_S.append( tf.constant(init_args.S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            inits_C.append( tf.truncated_normal(shape=[init_args.dims[l],1], mean=init_args.mu[l], stddev=init_args.std[l], dtype=tf.float64) )
    elif init_args.init_type  == 'data_init':
        X_train = init_args.X_train
        pass
    return (inits_C,inits_W,inits_S)

## Data sets
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape

## NN params
bn = True
if bn:
    phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON
else:
    phase_train = None

dims = [D,24,D_out]
dims = [D,24,24,D_out]
dims = [D,24,24,24,D_out]
#dims = [D,24,24,24,24,D_out]
mu = len(dims)*[0.0]
std = len(dims)*[0.1]
b_init = len(dims)*[0.1]
S_init = b_init
init_type = 'truncated_normal'
init_args = ns.FrozenNamespace(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init)
model = 'standard_nn'
model = 'summed_nn'
#model = 'hbf'
## Make Model
x = tf.placeholder(tf.float64, shape=[None, D], name='x-input') # M x D
nb_layers = len(dims)-1
nb_hidden_layers = nb_layers-1
print( '-----> Running model: %s. (nb_hidden_layers = %d, nb_layers = %d)' % (model,nb_hidden_layers,nb_layers) )
print( '-----> Units: %s)' % (dims) )
if model == 'standard_nn':
    tensorboard_data_dump = '/tmp/standard_nn_logs'
    (inits_C,inits_W,inits_b) = get_initilizations_standard_NN(init_args)
    with tf.name_scope("standardNN") as scope:
        nn = mtf.build_standard_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
        y = mtf.get_summation_layer(nn, inits_C[0])
elif model == 'summed_nn':
    tensorboard_data_dump = '/tmp/summed_nn_logs'
    (inits_C,inits_W,inits_b) = get_initilizations_summed_NN(init_args)
    with tf.name_scope("summNN") as scope:
        nn = mtf.build_summed_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
        y = nn
elif model == 'hbf':
    tensorboard_data_dump = '/tmp/summed_hbf_logs'
    (inits_C,inits_W,inits_S) = get_initilizations_summed_HBF(init_args)
    with tf.name_scope("summNN") as scope:
        hbf = mtf.build_summed_NN(x,dims,(inits_C,inits_W,inits_S),phase_train)
        y = hbf
    pass

## Output and Loss
y_ = tf.placeholder(tf.float64, shape=[None, D_out]) # (M x D)
with tf.name_scope("L2_loss") as scope:
    l2_loss = tf.reduce_mean(tf.square(y_-y))

## Training Step
optimization_alg = 'GD'
optimization_alg = 'Momentum'
optimization_alg = 'Adadelta'
optimization_alg = 'Adam'
#optimization_alg = 'Adagrad'
#optimization_alg = 'RMSProp'
with tf.name_scope("train") as scope:
    starter_learning_rate = 0.001
    decay_rate = 0.80
    decay_steps = 1000
    staircase = False
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

    # Passing global_step to minimize() will increment it at each step.
    if optimization_alg == 'GD':
        train_step = tf.GradientDescentOptimizer(starter_learning_rate).minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Momentum':
        train_step = tf.train.MomentumOptimizer(learning_rate=starter_learning_rate,momentum=0.9).minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adadelta':
        train_step = tf.train.AdadeltaOptimizer(learning_rate=starter_learning_rate, rho=0.95, epsilon=1e-08, name='Adadelta').minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adam':
        train_step = tf.train.AdamOptimizer(learning_rate=starter_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam').minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'Adagrad':
        train_step = tf.train.AdagradOptimizer(0.0001).minimize(l2_loss, global_step=global_step)
    elif optimization_alg == 'RMSProp':
        train_step = tf.train.RMSPropOptimizer(learning_rate=starter_learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, name='RMSProp').minimize(l2_loss, global_step=global_step)

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

start_time = time.time()
with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)

    sess.run( tf.initialize_all_variables() )
    steps = 120000
    M = 3000 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
        feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
        ## Train
        if i%50 == 0:
            train_result = sess.run([merged, l2_loss], feed_dict=feed_dict_train)
            summary_str_train = train_result[0]
            train_error = train_result[1]

            test_result = sess.run([merged, l2_loss], feed_dict=feed_dict_test)
            summary_str_test = test_result[0]
            test_error = test_result[1]

            writer.add_summary(summary_str_train, i)
            #print("step %d, training error %g"%(i, train_error))
            print("Model *%s%s*, step %d, training error %g, test error %g"%(model, nb_hidden_layers, i, train_error,test_error))
            print("Opt: %s, BN %s, After %d iteration:" % (optimization_alg,bn,i) )
        sess.run(train_step, feed_dict=feed_dict_batch)
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

seconds = (time.time() - start_time)
minutes = seconds/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
#winsound.Beep(Freq = 2500,Dur = 1000)
