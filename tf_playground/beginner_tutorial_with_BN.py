import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def batch_norm_layer(x,train_phase,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=True,
    reuse=None, # is this right?
    trainable=True,
    scope=scope_bn)

    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=False,
    reuse=True, # is this right?
    trainable=True,
    scope=scope_bn)

    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

def get_NN_layer(x, input_dim, output_dim, scope, train_phase):
    with tf.name_scope(scope+'vars'):
        W = tf.Variable(tf.truncated_normal(shape=[input_dim, output_dim], mean=0.0, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[output_dim]))
    with tf.name_scope(scope+'Z'):
        z = tf.matmul(x,W) + b
    with tf.name_scope(scope+'BN'):
        if train_phase is not None:
            z = batch_norm_layer(z,train_phase,scope+'BN_unit')
    with tf.name_scope(scope+'A'):
        a = tf.nn.relu(z) # (M x D1) = (M x D) * (D x D1)
    return a

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# placeholder for data
x = tf.placeholder(tf.float32, [None, 784])
# placeholder that turns BN during training or off during inference
train_phase = tf.placeholder(tf.bool, name='phase_train')
# variables for parameters
hiden_units = 25
layer1 = get_NN_layer(x, input_dim=784, output_dim=hiden_units, scope='layer1', train_phase=train_phase)
# create model
W_final = tf.Variable(tf.truncated_normal(shape=[hiden_units, 10], mean=0.0, stddev=0.1))
b_final = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(layer1, W_final) + b_final)

### training
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    steps = 3000
    for iter_step in xrange(steps):
        #feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # Collect model statistics
        if iter_step%1000 == 0:
            batch_xstrain, batch_xstrain = batch_xs, batch_ys #simualtes train data
            batch_xcv, batch_ycv = mnist.test.next_batch(5000) #simualtes CV data
            batch_xtest, batch_ytest = mnist.test.next_batch(5000) #simualtes test data
            # do inference
            train_error = sess.run(fetches=cross_entropy, feed_dict={x: batch_xs, y_:batch_ys, train_phase: False})
            cv_error = sess.run(fetches=cross_entropy, feed_dict={x: batch_xcv, y_:batch_ycv, train_phase: False})
            test_error = sess.run(fetches=cross_entropy, feed_dict={x: batch_xtest, y_:batch_ytest, train_phase: False})

            def do_stuff_with_errors(*args):
                print args
            do_stuff_with_errors(train_error, cv_error, test_error)
        # Run Train Step
        sess.run(fetches=train_step, feed_dict={x: batch_xs, y_:batch_ys, train_phase: True})
    # list of booleans indicating correct predictions
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, train_phase: False}))
