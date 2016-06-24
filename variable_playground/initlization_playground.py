import tensorflow as tf
import numpy as np

def generate_data(N_train_var=60000, N_cv_var=60000, N_test_var=60000, low_x_var=-2*np.pi, high_x_var=2*np.pi):
    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    low_x = low_x_var
    high_x = high_x_var
    # train
    N_train = N_train_var
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,1)
    Y_train = get_labels(X_train, np.zeros( (N_train,1) ) , f)
    # CV
    N_cv = N_cv_var
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,1)
    Y_cv = get_labels(X_cv, np.zeros( (N_cv,1) ), f)
    # test
    N_test = N_test_var
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,1)
    Y_test = get_labels(X_test, np.zeros( (N_test,1) ), f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def get_labels(X,Y,f):
    N_train = X.shape[0]
    for i in range(N_train):
        Y[i] = f(X[i])
    return Y


(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = generate_data()
(N_train,D) = X_train.shape
D1 = 10
(N_test,D_out) = Y_test.shape

# Create the model
x = tf.placeholder(tf.float32, [None, D], name="x-input")
#W = tf.Variable(tf.zeros([D,D1]), name="weights")
W = tf.Variable(  X_train[0:D1,:], name="W" )
#W = tf.Variable(tf.zeros([D,D1]), name="W")
b = tf.Variable(tf.zeros([D1], name="b"))
C = tf.Variable(tf.zeros([D1,D_out]), name="C")

# use a name scope to organize nodes in the graph visualizer
with tf.name_scope("Wx_b") as scope:
    z = tf.matmul(x,W) + b
with tf.name_scope("ReLu") as scope:
    a = tf.nn.relu(z, name=None)
with tf.name_scope("y") as scope:
    y = tf.matmul(a,C)

with tf.name_scope("L2_loss") as scope:
    l2_loss = tf.reduce_mean(tf.square(y_-y))

# Add summary ops to collect data
# w_hist = tf.histogram_summary("weights", W)
# b_hist = tf.histogram_summary("biases", b)
# y_hist = tf.histogram_summary("y", y)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None,10], name="y-input")
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    # Merge all the summaries and write them out to /tmp/mnist_logs
    #merged = tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)
    sess.run( tf.initialize_all_variables() )
    # Train the model, and feed in test data and record summaries every 10 steps
    steps = 2000
    M = 1000 #batch-size
    for i in range(steps):
        ## Create fake data for y = W.x + b where W = 2, b = 0
        #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
        feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
        ## Train
        if i%100 == 0:
            train_result = sess.run([merged, l2_loss], feed_dict=feed_dict_train)
            summary_str_train = train_result[0]
            train_error = train_result[1]

            # test_result = sess.run([merged, l2_loss], feed_dict=feed_dict_test)
            # summary_str_test = test_result[0]
            # test_error = test_result[1]

            writer.add_summary(summary_str_train, i)
            print("step %d, training accuracy %g"%(i, train_error))
            #print("step %d, training accuracy %g, test accuracy %g"%(i, train_error,test_error))
            print("After %d iteration:" % i)
        sess.run(train_step, feed_dict=feed_dict_batch)
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
