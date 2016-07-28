import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def add_batch_norm_layer(l, x, phase_train, n_out=1, scope='BN'):
    bn_layer = batch_norm_layer(x,phase_train,scope_bn=scope+l,trainable=True)
    return bn_layer

def batch_norm_layer(x,phase_train,scope_bn,trainable=True):
    print '======> official BN'
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=True,
    reuse=None, # is this right?
    trainable=trainable,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=False,
    reuse=True, # is this right?
    trainable=trainable,
    scope=scope_bn)
    z = tf.cond(phase_train, lambda: bn_train, lambda: bn_inference)
    return z

def get_NN_layer(l, x, dims, init, phase_train=None, scope="NNLayer"):
    init_W,init_b = init
    dim_input,dim_out = dims
    with tf.name_scope(scope+l):
        W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True, shape=[dim_input,dim_out])
        b = tf.get_variable(name='b'+l, dtype=tf.float64, initializer=init_b, regularizer=None, trainable=True)
        with tf.name_scope('Z'+l):
            z = tf.matmul(x,W) + b
            if phase_train is not None:
                z = add_batch_norm_layer(l, z, phase_train)
        with tf.name_scope('A'+l):
            a = tf.nn.relu(z) # (M x D1) = (M x D) * (D x D1)
            #a = tf.sigmoid(z)
    return a

def softmax_layer(l, x, dims, init):
    init_W,init_b = init
    dim_input,dim_out = dims
    with tf.name_scope('Z'+l):
        W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True, shape=[dim_input,dim_out])
        b = tf.get_variable(name='b'+l, dtype=tf.float64, initializer=init_b, regularizer=None, trainable=True)
        z = tf.matmul(x,W) + b
    with tf.name_scope('y'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y

###
###

def build_NN_two_hidden_layers(x, phase_train):
    ## first layer
    shape_layer1 = [784,50]
    init_W,init_b = tf.contrib.layers.xavier_initializer(dtype=tf.float64), tf.constant(0.1, shape=shape_layer1)
    A1 = get_NN_layer(l=1, x, dims=shape_layer1, init=(init_W,init_b), phase_train=phase_train, scope="NNLayer")
    ## second layer
    shape_layer2 = [50,49]
    init_W,init_b = (tf.contrib.layers.xavier_initializer(dtype=tf.float64), tf.constant(0.1, shape=shape_layer2)
    A2 = get_NN_layer(l=2, x, dims=shape_layer1, init=(init_W,init_b), phase_train=phase_train, scope="NNLayer")
    ## final layer
    y = softmax_layer(A2)
    return y

x = tf.placeholder(tf.float32, [None, 784])
y = build_NN_two_hidden_layers(x, phase_train)

cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )
# single training step opt.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#init op
init = tf.initialize_all_variables()
# launch model in a session
sess = tf.Session()
#run opt to initialize
sess.run(init)

# we'll run the training step 1000 times
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# list of booleans indicating correct predictions
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
