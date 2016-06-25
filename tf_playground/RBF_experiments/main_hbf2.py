import tensorflow as tf
import numpy as np
from f_1D_data import *

def get_Gaussian_layer(x,W,S,C,BN_layer = False, scale_bn = None, offset_bn = None, mu_bn = None, sig_bn = None):
    WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
    XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
    Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
    if BN_layer:
        mu = mu_bn
        sig = sig_bn
        offset = offset_bn
        scale = scale_bn
        Delta_tilde = tf.nn.batch_normalization(Delta_tilde, mu, sig, offset, scale, var_eps)
    beta = -1.0*tf.pow( 0.5*tf.div(1.0,S), 2)
    Z = beta * ( Delta_tilde ) # (M x D^(l))
    A = tf.exp(Z) # (M x D^(l))
    y_rbf = tf.matmul(A,C) # (M x 1) = (M x D^(l)) * (D^(l) x 1)
    return y_rbf

# launch interactive session
sess = tf.InteractiveSession()

(X_train, Y_train, X_test, Y_test) = get_data()
# nodes for the input images and target output classes
(N_train,D) = X_train.shape
D1 = 10
D2 = 10
(N_test,D_out) = Y_test.shape


x = tf.placeholder(tf.float32, shape=[None, D]) # M x D
# Variables Layer1
W1 = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
S1 = tf.Variable(tf.constant(0.0001, shape=[1])) # (1 x 1)
C1 = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1) ) # (D1 x 1)
BN_layer1 = False
if BN_layer1
    mean1 = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
    variance1 = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
    beta1 = tf.Variable(tf.constant(0.0, shape=[depth]))
    gamma1 = tf.Variable(tf.constant(1.0, shape=[depth]))
# Variables Layer2
W2 = tf.Variable( tf.truncated_normal([D,D2], mean=0.0, stddev=0.1) ) # (D x D1)
S2 = tf.Variable(tf.constant(0.0001, shape=[1])) # (1 x 1)
C2 = tf.Variable( tf.truncated_normal([D2,D_out], mean=0.0, stddev=0.1) ) # (D1 x 1)
BN_layer2 = False
if BN_layer2
    mean2 = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
    variance2 = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
    beta2 = tf.Variable(tf.constant(0.0, shape=[depth]))
    gamma2 = tf.Variable(tf.constant(1.0, shape=[depth]))

# make model
y_rbf1 = get_Gaussian_layer(x,W1,S1,C1,BN_layer1)
y_rbf2 = get_Gaussian_layer(y_rbf1,W2,S2,C2,BN_layer2)
y = y_rbf2
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)

#L2 loss/cost function sum((y_-y)**2)
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

sess = tf.Session()
init = tf.initialize_all_variables() #
sess.run(init)
steps = 8000
M = 10
for i in range(steps):
    batch = get_batch(X_train, Y_train, M)
    ## Train
    if i%2 == 0:
        train_error = sess.run(l2_loss, feed_dict={x:X_train, y_: Y_train})
        print("step %d, training accuracy %g"%(i, train_error))
        print("After %d iteration:" % i)
    batch_xs = batch[0]
    batch_ys = batch[0]
    feed = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed)
