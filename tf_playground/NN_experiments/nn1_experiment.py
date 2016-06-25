import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
from ..my_lib import *
#from my_lib import get_data
# import imp
# foo = imp.load_source('my_lib', '../my_lib/lib_building_blocks_nn_rbf.py')
# foo = imp.load_source('my_lib', '../my_lib/lib_building_blocks_nn_rbf.py')

(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data(file_name='f_1d_cos_no_noise_data')
(N_train,D) = X_train.shape
D1 = 72
(N_test,D_out) = Y_test.shape

def makeNNmodel(x,W,b,C):
    layer1 = summated_relu_layer(x,W,b,C,phase_train)
    y = layer1
    return f

## create BN variables
W = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
b = tf.Variable( tf.constant(0.1, shape=[D1]) ) # (D1 x 1)
C = tf.Variable( tf.truncated_normal([D1,D_out], mean=0.0, stddev=0.1) ) # (D1 x 1)
phase_train = tf.placeholder(tf.bool, name='phase_train')
#phase_train = None
x = tf.placeholder(tf.float32, name='input_image')
layer1 = makeNNmodel(x,W,b,C)
y = layer1
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)
l2_loss = tf.reduce_mean(tf.square(y_-y))
## TRAIN
def get_batch(X, Y, M):
    mini_batch_indices = np.random.randint(M,size=M)
    Xminibatch =  X[mini_batch_indices,:] # ( M x D^(0) )
    Yminibatch = Y[mini_batch_indices,:] # ( M x D^(L) )
    return (Xminibatch, Yminibatch)
# SGD alg
#train_step = tf.train.AdagradOptimizer(0.00001).minimize(l2_loss)
train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(l2_loss)
with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    steps = 8000
    M = 3000 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        (batch_xs, batch_ys) = get_batch(X_train, Y_train, M)
        ## Train
        if i%200 == 0:
            #train_error = sess.run(l2_loss, feed_dict={x:X_train, y_:Y_train, phase_train: False})
            train_error = sess.run(l2_loss, feed_dict={x:X_train, y_:Y_train})
            print("step %d, training accuracy %g"%(i, train_error))
            print("After %d iteration:" % i)
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, phase_train: True})
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
