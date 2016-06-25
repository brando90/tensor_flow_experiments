## run cmd to collect model: python test_hbf2_tensorboard.py --logdir=/tmp/hbf2_logs
## show board on browser run cmd: tensorboard --logdir=/tmp/hbf2_logs
## browser: http://localhost:6006/

import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
import my_lib.lib_building_blocks_nn_rbf as ml
import f_1D_data as data_lib
import time
#import winsound

def make_HBF1_model(x,W1,S1,C1,phase_train):
    with tf.name_scope("layer1") as scope:
        layer1 = ml.get_Gaussian_layer(x,W1,S1,C1,phase_train)
    y = layer1
    return y

(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data_lib.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
D1 = 24
(N_test,D_out) = Y_test.shape

x = tf.placeholder(tf.float32, shape=[None, D], name='x-input') # M x D
# Variables Layer1
#std = 1.5*np.pi
std = 0.1
W1 = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=std), name='W1') # (D x D1)
S1 = tf.Variable( tf.constant(100.0, shape=[1]), name='S1') # (1 x 1)
C1 = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1), name='C1') # (D1 x 1)
# BN layer
phase_train = None #BN OFF
#phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON

# make model
with tf.name_scope("HBF1") as scope:
    y = make_HBF1_model(x,W1,S1,C1,phase_train)
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
W1_hist = tf.histogram_summary("W1", W1)
S1_hist = tf.histogram_summary("S1", S1)
C1_hist = tf.histogram_summary("C1", C1)

with tf.name_scope("l2_loss") as scope:
  l2_hist = tf.histogram_summary("l2_loss", l2_loss)

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
    writer = tf.train.SummaryWriter("/tmp/hbf1_logs", sess.graph)

    sess.run( tf.initialize_all_variables() )
    steps = 500
    M = 100 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
        feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
        ## Train
        if i%50 == 0:
            train_result = sess.run([merged, l2_loss], feed_dict=feed_dict_train)
            test_result = sess.run([merged, l2_loss], feed_dict=feed_dict_test)

            summary_str_train = train_result[0]
            train_error = train_result[1]

            summary_str_test = test_result[0]
            test_error = test_result[1]

            writer.add_summary(summary_str_train, i)
            print("step %d, training accuracy %g, test accuracy %g"%(i, train_error,test_error))
            print("After %d iteration:" % i)
        sess.run(train_step, feed_dict=feed_dict_batch)
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

seconds = (time.time() - start_time)
minutes = seconds/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
#winsound.Beep(Freq = 2500,Dur = 1000)
