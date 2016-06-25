import tensorflow as tf
import pkg_1.module1 as mdl1

def g():
    mdl1.f()
    return tf.Variable([1,2,3])
