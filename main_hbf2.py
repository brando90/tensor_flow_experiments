import tensorflow as tf
import numpy as np
def get_labels(X,Y,f):
    N_train = X.shape[0]
    for i in range(N_train):
        Y[i] = f(X[i])
    return Y

def get_data(N_train_var=60000, N_test_var=60000, low_x_var=-np.pi, high_x_var=np.pi):
    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    low_x = low_x_var
    high_x = high_x_var
    # train
    N_train = N_train_var
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,1)
    Y_train = get_labels(X_train, np.zeros( (N_train,1) ) , f)
    # test
    N_test = N_test_var
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,1)
    Y_test = get_labels(X_test, np.zeros( (N_test,1) ), f)
    return (X_train, Y_train, X_test, Y_test)

def get_Gaussian_layer(x,W,S,C):
    WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
    XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
    Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
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

# make model
y_rbf1 = get_Gaussian_layer(x,W1,S1,C1)
y_rbf2 = get_Gaussian_layer(y_rbf1,W2,S2,C2)
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)

#L2 loss/cost function sum((y_-y)**2)
l2_loss = tf.reduce_mean(tf.square(y_-y))

# single training step opt
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(l2_loss)
train_step = tf.train.AdagradOptimizer(0.00001).minimize(l2_loss)

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
