import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder for data
x = tf.placeholder(tf.float32, [None, 784])
# variables for parameters

W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1) , name='W')
b = tf.Variable(tf.constant(0.1, shape=[10]), name='b')
Wx = tf.matmul(x, W)
#print Wx #Tensor("MatMul:0", shape=(?, 10), dtype=float32)
#print type(Wx) #<class 'tensorflow.python.framework.ops.Tensor'>

Wx_b = Wx + b
#print Wx_b #Tensor("add:0", shape=(?, 10), dtype=float32)
#print type(Wx_b) #<class 'tensorflow.python.framework.ops.Tensor'>

y = tf.nn.softmax(Wx_b)
### training
# new placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
#cost function
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )

# tensors have t.eval()
# ops have op.run()

c = tf.constant(1.0)
#print c #Tensor("Const_1:0", shape=(), dtype=float32)
#print type(c) #<class 'tensorflow.python.framework.ops.Tensor'>

# single training step opt
learning_rate = 1.0
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
opt = tf.train.GradientDescentOptimizer(learning_rate) ## <tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x10643c290>
gv = opt.compute_gradients(cross_entropy, [W,b])
print [ (g,v.name) for (g,v) in gv]
with tf.Session() as sess:
    epochs= 10
    for i in range(epochs):
        sess.run(tf.initialize_all_variables())
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # list of [ (Gradient, Variable) ]
        #(gradient, variable) = grads_and_vars[0]
        grad_vals = sess.run([g for (g,v) in gv], feed_dict={x: batch_xs, y_: batch_ys})
        print 'dJdW', grad_vals[0]
        print 'dJdb', grad_vals[1]
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# a = tf.constant(2.0)
# b = tf.constant(3.0)
# ab = tf.matmul(a, b)
# print ab
# print type(ab)

# single training step opt.
# learning_rate = 1
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate) ## <tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x10643c290>
#
# with tf.Session() as sess:
#     init = tf.initialize_all_variables()
#     sess.run(init)
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#
#     print optimizer # <tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x10643c290>
#     print type(optimizer) # <class 'tensorflow.python.training.gradient_descent.GradientDescentOptimizer'>
#
#     # list of [ (Gradient, Variable) ]
#     grads_and_vars = optimizer.compute_gradients(cross_entropy, [W])
#
#     print grads_and_vars #[(<tf.Tensor 'gradients_1/MatMul_grad/tuple/control_dependency_1:0' shape=(784, 10) dtype=float32>, <tensorflow.python.ops.variables.Variable object at 0x1109f7f50>)]
#     print type(grads_and_vars) # <type 'list'> = list of [ (Gradient, Variable) ]
#
#     # get variable and tensor for first variable
#     (gradient, variable) = grads_and_vars[0]
#     print variable is W
#
#     print gradient # Tensor("gradients_1/MatMul_grad/tuple/control_dependency_1:0", shape=(784, 10), dtype=float32)
#     print type(gradient) # <class 'tensorflow.python.framework.ops.Tensor'>
#
#     result = sess.run(gradient, feed_dict={x: batch_xs, y_: batch_ys})
#     print result

# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(1000):
#       batch_xs, batch_ys = mnist.train.next_batch(100)
#
#       grads_and_vars = optimizer.compute_gradients(cross_entropy, [W])
#       print type(train_step)
#       print grads_and_vars[0]
#       print sess.run( tf.identity(grads_and_vars[0][0]), feed_dict={x: batch_xs, y_: batch_ys})
#
#       sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
#     # list of booleans indicating correct predictions
#     correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
