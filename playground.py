import tensorflow as tf
import numpy as np

# Enter an interactive TensorFlow Session.
sess = tf.InteractiveSession()


x1 = tf.Variable( np.random.rand(10,1) )
x2 = tf.Variable( np.random.rand(1,10) )
x3 = tf.Variable( tf.constant(1.0, shape=[5,1]) )
x4 = tf.Variable( tf.constant(1.0, shape=[1,5]) )
x5 = tf.Variable( tf.constant(1.0, shape=[5]) )

A = tf.Variable( tf.constant(1.0, shape=[3,5]) )

# Initialize 'x' using the run() method of its initializer op.
# x1.initializer.run()
# x2.initializer.run()
# x3.initializer.run()
# x4.initializer.run()
# x5.initializer.run()

sess.run( tf.initialize_all_variables() )

# Run idenity op and print the result
# print( tf.identity(x1).eval() )
# print( tf.identity(x2).eval() )
# print( tf.identity(x3).eval() )
print( tf.identity(A + x3 ).eval() )
#print( tf.identity(x5).eval().shape )

sess.close()
