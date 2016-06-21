import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops

def batch_norm(x, D_l, phase_train, scope='bn'):
    """
    Batch normalization on vectors.
    Args:
        x:           vector [batch_size, D_l]
        D_l:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase, false testing/inference
        scope:       string, variable scope
    Return:
        bn_op:      batch-normalized op
    """
    with tf.variable_scope(scope):
        ## create BN variables
        #trainable params
        beta = tf.Variable(tf.constant(0.0, shape=[D_l]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[D_l]), name='gamma', trainable=True)
        #non-trainable params
        mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
        variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
        # testing & training utilities
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        # Maintains moving averages of variables by employing an exponential decay.
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def train_phase(batch_mean, batch_var):
            # during training use batch statistics
            mu = mean.assign(batch_mean)
            var = var.assign(batch_var)
            return mu, var
        def test_phase(batch_mean, batch_var):
            # during testing/inference use the population statistics that are computed using a moving average
            return ema.average(batch_mean), ema.average(batch_var)
        # testing & training utilities
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        mean, var = tf.cond(phase_train, train_phase, test_phase )
        bn_op = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return bn_op
