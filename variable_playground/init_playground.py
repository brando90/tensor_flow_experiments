import tensorflow as tf
import numpy as np

# Enter an interactive TensorFlow Session.
sess = tf.InteractiveSession()


x1 = tf.Variable([1.0, 2.0])
x2 = tf.Variable( np.random.rand(3,4) )

# Initialize 'x' using the run() method of its initializer op.
x1.initializer.run()
x2.initializer.run()

# Run idenity op and print the result
print( tf.identity(x1).eval() )
print( tf.identity(x2).eval() )

sess.close()
