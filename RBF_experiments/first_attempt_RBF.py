import tensorflow as tf
import numpy as np
from f_1D_data import *

(X_train, Y_train, X_test, Y_test) = get_data()
# nodes for the input images and target output classes
(N_train,D) = X_train.shape
D1 = 24
(N_test,D_out) = Y_test.shape


x = tf.placeholder(tf.float32, shape=[None, D]) # M x D
# Variable is a value that lives in TensorFlow's computation graph
W = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
S = tf.Variable(tf.constant(10.0, shape=[1])) # (1 x 1)
C = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1) ) # (D1 x 1)

# make model
WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
#Delta_tilde = 2.0*tf.matmul(x,W) - (WW + XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX)
#Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX)
beta = -1.0*tf.pow( 0.5*tf.div(1.0,S), 2)
Z = beta * ( Delta_tilde ) # (M x D^(l))
A = tf.exp(Z) # (M x D^(l))
y = tf.matmul(A,C) # (M x 1) = (M x D^(l)) * (D^(l) x 1)
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)

#L2 loss/cost function sum((y_-y)**2)
l2_loss = tf.reduce_mean(tf.square(y_-y))

# single training step opt
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(l2_loss)
#train_step = tf.train.AdagradOptimizer(0.00001).minimize(l2_loss)
train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.0009).minimize(l2_loss)

## TRAIN
def get_batch(X, Y, M):
    mini_batch_indices = np.random.randint(M,size=M)
    Xminibatch =  X_train[mini_batch_indices,:] # ( M x D^(0) )
    Yminibatch = Y_train[mini_batch_indices,:] # ( M x D^(L) )
    return (Xminibatch, Yminibatch)

sess = tf.Session()
init = tf.initialize_all_variables() #
sess.run(init)
steps = 8000
M = 100
for i in range(steps):
    ## Create fake data for y = W.x + b where W = 2, b = 0
    batch = get_batch(X_train, Y_train, M)
    ## Train
    if i%200 == 0:
        train_error = sess.run(l2_loss, feed_dict={x:X_train, y_: Y_train})
        print("step %d, training accuracy %g"%(i, train_error))
        print("After %d iteration:" % i)
    batch_xs = batch[0]
    batch_ys = batch[0]
    feed = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed)
