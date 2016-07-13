import tensorflow as tf

# Create some variables.
#tf.truncated_normal(shape=[m,n], mean=0.0, stddev=0.1)
A = tf.Variable(np.array([[1,2,3],[4,5,6]]), name='A')
x = tf.Variable(np.array([1,2,3]), name='x')
Ax = mt.matmul(A,x,name='Ax')
with tf.Session() as sess:


# Add ops to save and restore all the variables.
saver = tf.train.Saver()
