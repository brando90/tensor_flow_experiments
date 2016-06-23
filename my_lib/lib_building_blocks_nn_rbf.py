import numpy as np
import tensorflow as tf

def get_Gaussian_layer(x,W,S,C, phase_train=None):
    WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
    XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
    # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
    Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
    if phase_train is not None:
        Delta_tilde = standard_batch_norm(Delta_tilde, 1,phase_train)
    beta = 0.5*tf.pow(tf.div(1.0,S), 2)
    Z = beta * ( Delta_tilde ) # (M x D^(l))
    A = tf.exp(Z) # (M x D^(l))
    y_rbf = tf.matmul(A,C) # (M x 1) = (M x D^(l)) * (D^(l) x 1)
    return y_rbf

def get_summated_NN_layer(x,W,b,C,phase_train, scope='bn'):
    z1 = tf.matmul(x,W) + b # (M x D1)
    if phase_train is not None:
        z1 = standard_batch_norm(z1, 1, phase_train, scope)
    a = tf.nn.relu(z1) # (M x D1) = (M x D) * (D x D1)
    layer = tf.matmul(a,C)
    return layer

def standard_batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Vector
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
