import tensorflow as tf
import numpy as np
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# launch interactive session
sess = tf.InteractiveSession()
# nodes for the input images and target output classes
D1 = 10
x = tf.placeholder(tf.float32, shape=[None, 784]) # M x D

# Variable is a value that lives in TensorFlow's computation graph
W = tf.Variable( tf.truncated_normal([784,D1], mean=0.0, stddev=0.1) ) # (D x D1)
b = tf.Variable(tf.constant(0.1, shape=[D1])) # (D1 x 1)
C = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1) ) # (D1 x 1)

# make model
z1 = tf.matmul(x,W) + b # M x D1
a = tf.nn.relu( tf.matmul(x,W) + b) # (M x D1) = (M x D) * (D x D1)
y = tf.matmul(a,C) # (M x 1) = (M x D1) * (D1 x 1)
y_ = tf.placeholder(tf.float32, shape=[None, 784]) # (M x D)

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
