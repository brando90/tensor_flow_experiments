import tensorflow as tf
# download and install the MNIST data automatically
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder for data
x = tf.Variable(1.0)
b = tf.placeholder(tf.float32)
xx_b = 0.5*(x-b)*(x-b)
y=xx_b

# single training step opt
learning_rate = 1
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
opt = tf.train.GradientDescentOptimizer(learning_rate) ## <tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x10643c290>
grads = opt.compute_gradients(y) # List[ (gradient,variable) ]

with tf.Session() as sess:
    b_val = 0.7
    sess.run(tf.initialize_all_variables())
    grad_vals = sess.run([grad[0] for grad in grads], feed_dict={b: b_val})
    print grad_vals
