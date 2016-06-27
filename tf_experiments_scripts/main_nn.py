## run cmd to collect model: python main_nn.py --logdir=/tmp/nn_logs
## show board on browser run cmd: tensorboard --logdir=/tmp/nn_logs
## browser: http://localhost:6006/

import numpy as np
import tensorflow as tf

import my_tf_pkg as mtf
#from tensorflow.python import control_flow_ops
import time

import namespaces as ns

#import winsound

# def get_initilizations(**kwargs):
#     if kwargs['init_type'] == 'truncated_normal':
#         dims = kwargs['dims']
#         inits_W = [None]
#         for l in range(1,):
#             inits_W.append(tf.truncated_normal(shape=dims[l-1], dims[l], mean=kwargs['mean'], stddev=kwargs['stddev']))
#             init_b.append(tf.constant(0.1, shape=[dims[l]]))
#     elif kwargs['init_type']  == 'data_init':
#         pass
#     return inits_W,init_b

def get_initilizations(init_args):
    if init_args.init_type == 'truncated_normal':
        inits_W = [None]
        inits_b = [None]
        nb_hidden_layers = len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[init_args.dims[l-1],init_args.dims[l]], mean=init_args.mu[l], stddev=init_args.std[l], dtype=tf.float64) )
            inits_b.append( tf.constant(init_args.b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        l = len(init_args.dims)-1
        inits_C = [ tf.truncated_normal(dtype=tf.float64, shape=[init_args.dims[l-1],init_args.dims[l]], mean=init_args.mu, stddev=init_args.std) ]
    elif init_args.init_type  == 'data_init':
        X_train = init_args.X_train
        pass
    return (inits_C,inits_W,inits_b)

## Data sets
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape

## NN params
tensorboard_data_dump = '/tmp/nn_logs'
phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON
phase_train = None
dims = [D,10,D_out]
dims = [D,10,10,10,D_out]
mu = len(dims)*[0.0]
std = len(dims)*[0.1]
b_init = len(dims)*[0.1]
init_type = 'truncated_normal'
init_args = ns.FrozenNamespace(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init)
(inits_C,inits_W,init_b) = get_initilizations(init_args)

## Make Model
x = tf.placeholder(tf.float64, shape=[None, D], name='x-input') # M x D
with tf.name_scope("NN") as scope:
    nn = mtf.build_NN(x,dims,(inits_C,inits_W,init_b),phase_train)
    y = mtf.get_summation_layer(nn, inits_C[0])

## Output and Loss
y_ = tf.placeholder(tf.float64, shape=[None, D_out]) # (M x D)
with tf.name_scope("L2_loss") as scope:
    l2_loss = tf.reduce_mean(tf.square(y_-y))

## Training Step
with tf.name_scope("train") as scope:
    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(l2_loss)
    #train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.6).minimize(l2_loss)
    #train_step = tf.train.AdadeltaOptimizer.(learning_rate=0.001, rho=0.95, epsilon=1e-08, name='Adadelta').minimize(l2_loss)
    train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam').minimize(l2_loss)
    #train_step = tf.train.AdagradOptimizer(0.0001).minimize(l2_loss)
    #train_step = tf.train.RMSPropOptimizer.(learning_rate=0.01, decay=0.9, momentum=0.0, epsilon=1e-10, name='RMSProp').minimize(l2_loss)


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
    steps = 2000
    M = 100 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
        feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
        ## Train
        if i%100 == 0:
            train_result = sess.run([merged, l2_loss], feed_dict=feed_dict_train)
            summary_str_train = train_result[0]
            train_error = train_result[1]

            # test_result = sess.run([merged, l2_loss], feed_dict=feed_dict_test)
            # summary_str_test = test_result[0]
            # test_error = test_result[1]

            writer.add_summary(summary_str_train, i)
            print("step %d, training accuracy %g"%(i, train_error))
            #print("step %d, training accuracy %g, test accuracy %g"%(i, train_error,test_error))
            print("After %d iteration:" % i)
        sess.run(train_step, feed_dict=feed_dict_batch)
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

seconds = (time.time() - start_time)
minutes = seconds/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
#winsound.Beep(Freq = 2500,Dur = 1000)
