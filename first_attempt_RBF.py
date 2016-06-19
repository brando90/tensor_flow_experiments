import tensorflow as tf
import numpy as np

def get_data(N_train_var=60000, N_test_var=60000,low_x_var=-np.pi, high_x_var=np.pi)
    def get_labels(X,Y,f):
        for i in range(X)
            Y[i] = f(x[i])
        return Y

    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    low_x = low_x_var
    high_x = high_x_var
    # train
    N_train = N_train_var
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,1)
    Y_train = get_labels(X_train,np.zeros(N_train,1),f)
    # test
    N_test = N_test_var
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,1)
    Y_test = get_labels(X_test,np.zeros(N_test,1),f)
    return (X_train, Y_train, X_test, Y_test)

# launch interactive session
sess = tf.InteractiveSession()

(X_train, Y_train, X_test, Y_test) = get_data()
# nodes for the input images and target output classes
(N_train,D) = X_train.shape
D1 = 10
(N_test,D_out) = Y_test.shape


x = tf.placeholder(tf.float32, shape=[None, D]) # M x D
# Variable is a value that lives in TensorFlow's computation graph
W = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
S = tf.Variable(tf.constant(0.001, shape=[1])) # (1 x 1)
C = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1) ) # (D1 x 1)

# make model
WW =  tf.reduce_sum(W*W, reduction_indices=0) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
XX =  tf.reduce_sum(x*x, reduction_indices=1) # (M x 1) = sum( (M x D^(l-1)), 1 )
Delta_tilde = 2*tf.matmul(x,W) - (WW + XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
beta = -1*tf.pow( 0.5*tf.div(1,S), 2)
Z = beta * ( Delta_tilde ) # (M x D^(l))
A = tf.exp(Z) # (M x D^(l))
y = tf.matmul(A,C) # (M x 1) = (M x D^(l)) * (D^(l) x 1)
y_ = tf.placeholder(tf.float32, shape=[None, D_out]) # (M x D)

#L2 loss/cost function sum((y_-y)**2)
l2_loss = tf.reduce_mean(tf.square(y_-y))

# single training step opt
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(l2_loss)

## TRAIN
sess = tf.Session()
init = tf.initialize_all_variables() #
sess.run(init)
steps = 5000
M = 50
for i in range(steps):
    # Create fake data for y = W.x + b where W = 2, b = 0
    #xs = np.array([[i]])
    #ys = np.array([[2*i + 1]])
    batch = mnist.train.next_batch(M)
    # Train
    if i%100 == 0:
        #train_accuracy = l2_loss.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_batch = mnist.train.next_batch(50000)
        train_error = sess.run(l2_loss, feed_dict={x:train_batch[0], y_: train_batch[0]})
        #train_accuracy =  feed_dict={x:batch[0], y_: batch[1]})
        print("step %d, training accuracy %g"%(i, train_error))
        print("After %d iteration:" % i)
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #feed = { x: xs, y_: ys }
    #train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    batch_xs = batch[0]
    batch_ys = batch[0]
    feed = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed)
