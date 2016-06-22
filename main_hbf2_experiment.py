import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
import my_lib.lib_building_blocks_nn_rbf as ml
import f_1D_data as data_lib

def make_RBF2_model(x,W1,S1,C1,W2,S2,C2,phase_train):
    layer1 = ml.get_Gaussian_layer(x,W1,S1,C1,phase_train)
    layer2 = ml.get_Gaussian_layer(layer1,W2,S2,C2,phase_train)
    y = layer2
    return y

(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = data_lib.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
D1 = 72
D2 = 72
(N_test,D_out) = Y_test.shape

x = tf.placeholder(tf.float32, shape=[None, D]) # M x D
# Variables Layer1
W1 = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
S1 = tf.Variable(tf.constant(0.0001, shape=[1])) # (1 x 1)
C1 = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1) ) # (D1 x 1)
# Variables Layer2
W2 = tf.Variable( tf.truncated_normal([D,D2], mean=0.0, stddev=0.1) ) # (D x D1)
S2 = tf.Variable(tf.constant(0.0001, shape=[1])) # (1 x 1)
C2 = tf.Variable( tf.truncated_normal([D2,D_out], mean=0.0, stddev=0.1) ) # (D1 x 1)
# BN layer
#phase_train = None #BN OFF
phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON

# make model
y = make_RBF2_model(x,W1,S1,C1,W2,S2,C2,phase_train)
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)
l2_loss = tf.reduce_mean(tf.square(y_-y))

# single training step opt
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(l2_loss)
train_step = tf.train.AdagradOptimizer(0.00001).minimize(l2_loss)

## TRAIN
def get_batch(X, Y, M):
    mini_batch_indices = np.random.randint(M,size=M)
    Xminibatch =  X[mini_batch_indices,:] # ( M x D^(0) )
    Yminibatch = Y[mini_batch_indices,:] # ( M x D^(L) )
    return (Xminibatch, Yminibatch)
with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    steps = 8000
    M = 3000 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        (batch_xs, batch_ys) = get_batch(X_train, Y_train, M)
        ## Train
        if i%200 == 0:
            train_error = sess.run(l2_loss, feed_dict={x:X_train, y_:Y_train, phase_train: False})
            #train_error = sess.run(l2_loss, feed_dict={x:X_train, y_:Y_train})
            print("step %d, training accuracy %g"%(i, train_error))
            print("After %d iteration:" % i)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, phase_train: True})
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
