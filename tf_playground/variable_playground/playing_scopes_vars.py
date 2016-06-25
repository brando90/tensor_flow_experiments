import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops

def scope_test_with_constructor(n_out=1,scope='test_scope'):
    with tf.variable_scope(scope):
        var1 = tf.Variable(tf.constant(0.0, shape=[n_out]), name='v1', trainable=True)
        var2 = tf.Variable(tf.constant(1.0, shape=[n_out]), name='v2', trainable=True)
    return var1, var2

def scope_test_get_variable(n_out=1,scope='test_scope', init=tf.random_normal_initializer()):
    with tf.variable_scope(scope):
        var1 = tf.get_variable("v1", [n_out], initializer=init)
        var2 = tf.get_variable("v2", [n_out], initializer=init)
    return var1, var2

# No error but returns different names for the variables
(var1, var2) = scope_test_with_constructor()
(v1, v2) = scope_test_with_constructor()
(vv1, vv2) = scope_test_with_constructor()

# ValueError: Variable test_scope/v1 already exists, disallowed. Did you mean to set reuse=True in VarScope?
# (var1, var2) = scope_test_get_variable()
# (v1, v2) = scope_test_get_variable()
# (vv1, vv2) = scope_test_get_variable()

print var1 == v1
print var2 == v2
print vv1 == vv2

print var1.name
print v1.name
print var2.name
print v2.name
print vv1.name
print vv2.name
