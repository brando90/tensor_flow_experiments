import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
from f_1D_data import *

def batch_norm(x, beta, gamma, mean, var, phase_train, bn_eps=1e-3):
    """
    Batch normalization on vectors.
    Args:
        x:           vector [batch_size, D_l]
        beta:
        gamma:
        mean:
        var
        phase_train:
        bn_eps
    Return:
        bn_op:      batch-normalized op
    """
    batch_mean, batch_var = tf.nn.moments(x, axes=[0], name='moments')
    def update_bn_moments_train():
        # during training use batch statistics
        mu_out = mean.assign(batch_mean)
        var_out = var.assign(batch_var)
        return mu_out, var_out
    def update_bn_moments_test():
        # during testing/inference use the population statistics that are computed using a moving average
        return ema.average(batch_mean), ema.average(batch_var) # returns the Variable holding the (exp mov) average of batch_var.
    # testing & training utilities
    print phase_train == None
    mean, var = tf.cond(phase_train, update_bn_moments_train, update_bn_moments_test )
    bn_op = tf.nn.batch_normalization(x, mean, var, beta, gamma, bn_eps)
    return bn_op

def summated_relu_layer(x,W,b,C,bn_layer=False,beta=None,gamma=None,mean=None,var=None,phase_train=None):
    z1 = tf.matmul(x,W) + b # (M x D1)
    if bn_layer:
        z1 = batch_norm(z1, beta, gamma, mean, var, phase_train)
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
beta = tf.Variable(tf.constant(0.0, shape=[D1]), name='beta', trainable=True) #trainable params
gamma = tf.Variable(tf.constant(1.0, shape=[D1]), name='gamma', trainable=True) #trainable params
mean = tf.Variable(tf.constant(0.0, shape=[D1]), trainable=False) #non-trainable params
var = tf.Variable(tf.constant(1.0, shape=[D1]), trainable=False) #non-trainable params
ema = tf.train.ExponentialMovingAverage(decay=0.5) # Maintains moving averages of variables by employing an exponential decay.
phase_train = tf.placeholder(tf.bool, name='phase_train')
#
C = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1) ) # (D1 x 1)
# make model
x = tf.placeholder(tf.float32, name='input_image')
layer1 = summated_relu_layer(x,W,b,C,True,beta,gamma,mean,var,phase_train)
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
train_step = tf.train.AdagradOptimizer(0.00001).minimize(l2_loss)
with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    steps = 8000
    M = 100 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        (batch_xs, batch_ys) = get_batch(X_train, Y_train, M)
        ## Train
        if i%200 == 0:
            train_error = sess.run(l2_loss, feed_dict={x:X_train, y_:Y_train, phase_train: False})
            print("step %d, training accuracy %g"%(i, train_error))
            print("After %d iteration:" % i)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, phase_train: True})
