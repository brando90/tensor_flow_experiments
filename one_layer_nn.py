import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# launch interactive session
sess = tf.InteractiveSession()
# nodes for the input images and target output classes
D1 = 10
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, D1])

# Variable is a value that lives in TensorFlow's computation graph
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
C = tf.tf.Variable(tf.zeros([10,1]))

# init variables
sess.run(tf.initialize_all_variables())
# make model
a = tf.nn.relu( tf.matmul(x,W) + b) # M  x D1
y = tf.matmul(a,C)

#L2 loss
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )
l2_loss =
