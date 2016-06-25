## run cmd to collect model: python test_hbf1_tensorboard.py --logdir=/tmp/nn1_logs
## show board on browser run cmd: tensorboard --logdir=/tmp/nn1_logs
## browser: http://localhost:6006/

import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
import my_lib.lib_building_blocks_nn_rbf as ml
import f_1D_data as data_lib
import time
#import winsound

tensorboard_data_loc = "nn1_logs"

(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data_lib.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape
dimensions_list = [D,10,D_out]
nb_layers = 1

x = tf.placeholder(tf.float32, shape=[None, D], name='x-input') # M x D
# BN layer
#phase_train = None #BN OFF
phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON

### make model
with tf.name_scope( "NN"+str(nb_layers) ) as scope:
    nn = ml.build_NN(x, dimensions_list)

y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)
with tf.name_scope("L2_loss") as scope:
    l2_loss = tf.reduce_mean(tf.square(y_-y))

# single training step opt
with tf.name_scope("train") as scope:
    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(l2_loss)
    train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.6).minimize(l2_loss)
    #train_step = tf.train.AdadeltaOptimizer.(learning_rate=0.001, rho=0.95, epsilon=1e-08, name='Adadelta').minimize(l2_loss)
    #train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam').minimize(l2_loss)
    #train_step = tf.train.AdagradOptimizer(0.0001).minimize(l2_loss)
    #train_step = tf.train.RMSPropOptimizer.(learning_rate=0.01, decay=0.9, momentum=0.0, epsilon=1e-10, name='RMSProp').minimize(l2_loss)

## Add summary ops to collect data
#W1_hist = tf.histogram_summary("W1_hist", W1)
#W1_scalar_summary = tf.scalar_summary("W1_scalar", W1)
#W1_hist = tf.histogram_summary("W1", W1)

#S1_hist = tf.histogram_summary("S1_hist", S1)
#S1_scalar_summary = tf.scalar_summary("S1_scalar", S1)
#S1_scalar_summary = tf.scalar_summary("S1", S1)

#C1_hist = tf.histogram_summary("C1_hist", C1)
#C1_scalar_summary = tf.scalar_summary("C1_scalar", C1)
#C1_hist = tf.histogram_summary("C1", C1)

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
    writer = tf.train.SummaryWriter(tensorboard_data_loc, sess.graph)

    sess.run( tf.initialize_all_variables() )
    steps = 270000
    M = 1000 #batch-size
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
