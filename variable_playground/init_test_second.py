import tensorflow as tf
import numpy as np

# Enter an interactive TensorFlow Session.
sess = tf.InteractiveSession()

X = np.random.rand(10,3)

x = tf.Variable( X[0:4,:] )

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Run idenity op and print the result
print( tf.identity(x).eval() )

sess.close()
