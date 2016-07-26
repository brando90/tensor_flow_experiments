import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(5)

print mnist
print mnist.train.images.shape
print mnist.validation.images.shape
print mnist.test.images.shape
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
