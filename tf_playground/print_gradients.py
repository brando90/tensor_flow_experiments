import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder for data
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))
Wx = tf.matmul(x, W)
Wx_b = Wx + b
y = tf.nn.softmax(Wx_b)
# new placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
#cost function
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )
# single training step opt
learning_rate = 1.0
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate) ## <tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x10643c290>
with tf.Session() as sess:
    for i in range(1000):
        sess.run(tf.initialize_all_variables())
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # list of [ (Gradient, Variable) ]
        grads_and_vars = optimizer.compute_gradients(cross_entropy, [W])
        (gradient, variable) = grads_and_vars[0]
        result = sess.run(gradient, feed_dict={x: batch_xs, y_: batch_ys})

# with tf.Session() as sess:
#     for i in range(1000):
#         sess.run(tf.initialize_all_variables())
#         batch_xs, batch_ys = mnist.train.next_batch(100)
#
#         print optimizer # <tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x10643c290>
#         print type(optimizer) # <class 'tensorflow.python.training.gradient_descent.GradientDescentOptimizer'>
#
#         # list of [ (Gradient, Variable) ]
#         grads_and_vars = optimizer.compute_gradients(cross_entropy, [W])
#
#         print grads_and_vars #[(<tf.Tensor 'gradients_1/MatMul_grad/tuple/control_dependency_1:0' shape=(784, 10) dtype=float32>, <tensorflow.python.ops.variables.Variable object at 0x1109f7f50>)]
#         print type(grads_and_vars) # <type 'list'> = list of [ (Gradient, Variable) ]
#
#         # get variable and tensor for first variable
#         (gradient, variable) = grads_and_vars[0]
#         print variable is W
#
#         print gradient # Tensor("gradients_1/MatMul_grad/tuple/control_dependency_1:0", shape=(784, 10), dtype=float32)
#         print type(gradient) # <class 'tensorflow.python.framework.ops.Tensor'>
#
#         result = sess.run(gradient, feed_dict={x: batch_xs, y_: batch_ys})
#         print result
