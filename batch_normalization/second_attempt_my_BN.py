import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops

def batch_norm(x, beta, gamma, mean, var, phase_train, bn_eps=1e-3):
    """
    Batch normalization on vectors.
    Args:
        x:           vector [batch_size, D_l]
        beta:
        gamma:
        mean:
        var
        phase_train:
        bn_eps
    Return:
        bn_op:      batch-normalized op
    """
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    def update_bn_moments_train(batch_mean, batch_var):
        # during training use batch statistics
        mu = mean.assign(batch_mean)
        var = var.assign(batch_var)
        return mu, var
    def update_bn_moments_test(batch_mean, batch_var):
        # during testing/inference use the population statistics that are computed using a moving average
        return ema.average(batch_mean), ema.average(batch_var) # returns the Variable holding the (exp mov) average of batch_var.
    # testing & training utilities
    mean, var = tf.cond(phase_train, update_bn_moments_train, update_bn_moments_test )
    bn_op = tf.nn.batch_normalization(x, mean, var, beta, gamma, bn_eps)
    return bn_op
