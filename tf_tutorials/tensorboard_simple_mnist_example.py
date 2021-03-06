## run command to collect model: python tensorboard_simple_mnist_example.py --logdir=/tmp/mnist_logs
## tensorboard --logdir=/tmp/mnist_logs
## run http://localhost:6006/

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name="x-input")
W = tf.Variable(tf.zeros([784,10]), name="weights")
b = tf.Variable(tf.zeros([10], name="bias"))

# use a name scope to organize nodes in the graph visualizer
with tf.name_scope("Wx_b") as scope:
  y = tf.nn.softmax(tf.matmul(x,W) + b)

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None,10], name="y-input")
# More name scopes will clean up the graph representation
with tf.name_scope("xent") as scope:
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("test") as scope:
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)

with tf.Session() as sess:
    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)
    tf.initialize_all_variables().run()
    # Train the model, and feed in test data and record summaries every 10 steps
    for i in range(1000):
      if i % 10 == 0:  # Record summary data, and the accuracy
        feed = {x: mnist.test.images, y_: mnist.test.labels}
        result = sess.run([merged, accuracy], feed_dict=feed)
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, i)
        print("Accuracy at step %s: %s" % (i, acc))
      else:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed = {x: batch_xs, y_: batch_ys}
        sess.run(train_step, feed_dict=feed)

    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
