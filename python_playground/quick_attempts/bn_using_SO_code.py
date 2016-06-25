import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
from f_1D_data import *

def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def summated_relu_layer(x,W,b,C,phase_train, scope='bn'):
    z1 = tf.matmul(x,W) + b # (M x D1)
    if phase_train is not None:
        z1 = batch_norm(z1, 1, phase_train, scope)
    a = tf.nn.relu(z1) # (M x D1) = (M x D) * (D x D1)
    layer = tf.matmul(a,C)
    return layer

##################
##################
##################

(X_train, Y_train, X_test, Y_test) = get_data()
(N_train,D) = X_train.shape
D1 = 72
(N_test,D_out) = Y_test.shape

## create BN variables
#
W = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
b = tf.Variable(tf.constant(0.1, shape=[D1])) # (D1 x 1)
#
# beta = tf.Variable(tf.constant(0.0, shape=[D1]), name='beta', trainable=True) #trainable params
# gamma = tf.Variable(tf.constant(1.0, shape=[D1]), name='gamma', trainable=True) #trainable params
# mean = tf.Variable(tf.constant(0.0, shape=[D1]), trainable=False) #non-trainable params
# var = tf.Variable(tf.constant(1.0, shape=[D1]), trainable=False) #non-trainable params
# ema = tf.train.ExponentialMovingAverage(decay=0.5) # Maintains moving averages of variables by employing an exponential decay.
phase_train = tf.placeholder(tf.bool, name='phase_train')
phase_train = None
#
C = tf.Variable( tf.truncated_normal([D1,D_out], mean=0.0, stddev=0.1) ) # (D1 x 1)
# make model
x = tf.placeholder(tf.float32, name='input_image')
layer1 = summated_relu_layer(x,W,b,C,phase_train)
y = layer1
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)
#L2 loss/cost function sum((y_-y)**2)
l2_loss = tf.reduce_mean(tf.square(y_-y))
#
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
