import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
import my_lib.lib_building_blocks_nn_rbf as ml
import f_1D_data as data_lib
import time
#import winsound

def make_HBF2_model(x,W1,S1,C1,W2,S2,C2,phase_train):
    layer1 = ml.get_Gaussian_layer(x,W1,S1,C1,phase_train)
    layer2 = ml.get_Gaussian_layer(layer1,W2,S2,C2,phase_train)
    y = layer2
    return y

(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data_lib.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
print np.amax(X_train)
print np.amin(X_train)
D1 = 48
D2 = 48
(N_test,D_out) = Y_test.shape

x = tf.placeholder(tf.float32, shape=[None, D]) # M x D
# Variables Layer1
#std = 1.5*np.pi
std = 0.1
W1 = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=std) ) # (D x D1)
S1 = tf.Variable(tf.constant(25.0, shape=[1])) # (1 x 1)
C1 = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1) ) # (D1 x 1)
# Variables Layer2
W2 = tf.Variable( tf.truncated_normal([D,D2], mean=0.0, stddev=std) ) # (D x D1)
S2 = tf.Variable(tf.constant(25.0, shape=[1])) # (1 x 1)
C2 = tf.Variable( tf.truncated_normal([D2,D_out], mean=0.0, stddev=0.1) ) # (D1 x 1)
# BN layer
#phase_train = None #BN OFF
phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON

# make model
y = make_HBF2_model(x,W1,S1,C1,W2,S2,C2,phase_train)
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)
l2_loss = tf.reduce_mean(tf.square(y_-y))

# single training step opt
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(l2_loss)
#train_step = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.6).minimize(l2_loss)
#train_step = tf.train.AdadeltaOptimizer.(learning_rate=0.001, rho=0.95, epsilon=1e-08, name='Adadelta').minimize(l2_loss)
train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam').minimize(l2_loss)
#train_step = tf.train.AdagradOptimizer(0.0001).minimize(l2_loss)
#train_step = tf.train.RMSPropOptimizer.(learning_rate=0.01, decay=0.9, momentum=0.0, epsilon=1e-10, name='RMSProp').minimize(l2_loss)

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
    sess.run( tf.initialize_all_variables() )
    steps = 10
    M = 2000 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
        feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
        ## Train
        if i%100 == 0:
            train_error = sess.run(l2_loss, feed_dict=feed_dict_train)
            test_error = sess.run(l2_loss, feed_dict=feed_dict_test)
            #train_error = sess.run(l2_loss, feed_dict={x:X_train, y_:Y_train})
            print("step %d, training accuracy %g, test accuracy %g"%(i, train_error,test_error))
            print("After %d iteration:" % i)
        sess.run(train_step, feed_dict=feed_dict_batch)
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#winsound.Beep(Freq = 2500,Dur = 1000)
seconds = (time.time() - start_time)
minutes = seconds/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
