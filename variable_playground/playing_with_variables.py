import numpy as np
import tensorflow as tf

D=1
v1 = tf.Variable(tf.constant(0.0, shape=[D]), name='v1', trainable=True)
v2 = tf.Variable(tf.constant(1.0, shape=[D]), name='v1', trainable=True)

v = tf.get_variable("v1", [1])
#v1 = tf.get_variable("v", [1])
print v1.name
print v2.name
print v.name

print v1 == v
assert v1 == v
