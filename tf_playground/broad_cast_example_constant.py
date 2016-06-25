import tensorflow as tf
import numpy as np

M = 5
D = 2
D1 = 3
x = tf.constant( np.random.rand(M,D) )
W = tf.constant( np.random.rand(D,D1) )

#x = tf.placeholder(tf.float32, shape=[None, D]) # M x D
# Variable is a value that lives in TensorFlow's computation graph
#W = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=0.1) ) # (D x D1)
WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
#Delta_tilde = 2.0*tf.matmul(x,W) - (WW + XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
P1 = tf.add(WW, XX)
P2 = 2.0*tf.matmul(x,W)
Delta_tilde = P1 - P2

y_ = tf.placeholder(tf.float32, shape=[None, 1])

## Launch the default graph.
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
result = sess.run(Delta_tilde)
print(result)
print(type(result))
sess.close()

## The Session closes automatically at the end of the with block.
# with tf.Session() as sess:
#   result = sess.run(product)
#   print(result)
#   print(type(result))
