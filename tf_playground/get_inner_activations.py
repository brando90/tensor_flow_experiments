import tensorflow as tf
import numpy as np
import my_tf_pkg as mtf

N = 20
D = 3
D1 = 4
D2 = 5
X_train = np.random.rand(N,D)
X_train = 2*X_train + 3
x = tf.placeholder(tf.float64, shape=[None,D], name='x-input')

mu1 = 0.0
mu2 = mu1
muC = mu2
std1 = 0.1
std2 = std1
stdC = std2
const1 = 0.1
const2 = const1

W1 = tf.Variable(tf.truncated_normal(shape=[D,D1], mean=mu1, stddev=std1, dtype=tf.float64))
b1 = tf.Variable(tf.constant(const1,shape=[D1],dtype=tf.float64))
W2 = tf.Variable(tf.truncated_normal(shape=[D,D1], mean=mu2, stddev=std2, dtype=tf.float64))
b2 = tf.Variable(tf.constant(const2,shape=[D2],dtype=tf.float64))
C = tf.Variable(tf.truncated_normal(shape=[D,D1], mean=muC, stddev=stdC, dtype=tf.float64))

z1 = tf.matmul(x,W1) + b1
a1 = tf.relu(z)
z2 = tf.matmul(a1,W2) + b2
a2 = tf.rely(z2)

with tf.Session() sess:
