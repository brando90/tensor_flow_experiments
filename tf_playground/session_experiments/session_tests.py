import numpy as np
import tensorflow as tf

M = 4
D = 2
D1 = 3
x = tf.placeholder(tf.float32, shape=[M, D]) # M x D
W = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
b = tf.Variable( tf.constant(0.1, shape=[D1]) ) # (D1 x 1)
inner_product = tf.matmul(x,W) + b # M x D1
with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    x_val = np.random.rand(M,D)
    ans = sess.run(inner_product, feed_dict={x: x_val})
    print ans