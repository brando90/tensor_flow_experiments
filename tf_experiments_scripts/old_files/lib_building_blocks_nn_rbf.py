import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import pdb

def get_Gaussian_layer(x,W,S,C, phase_train=None):
    with tf.name_scope("Z-pre_acts") as scope:
        WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
        XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
        # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
        Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
        if phase_train is not None:
            Delta_tilde = standard_batch_norm(Delta_tilde, 1,phase_train)
        beta = 0.5*tf.pow(tf.div(1.0,S), 2)
        Z = beta * ( Delta_tilde ) # (M x D^(l))
    with tf.name_scope("A-acts") as scope:
        A = tf.exp(Z) # (M x D^(l))
    with tf.name_scope("Sum-C_iA_i-sum_acts") as scope:
        y_rbf = tf.matmul(A,C) # (M x 1) = (M x D^(l)) * (D^(l) x 1)
    return y_rbf

def get_summated_NN_layer(x,W,b,C,phase_train=None, scope='SumRelu'):
    z1 = tf.matmul(x,W) + b # (M x D1)
    if phase_train is not None:
        z1 = standard_batch_norm(z1, 1, phase_train, scope)
    a = tf.nn.relu(z1) # (M x D1) = (M x D) * (D x D1)
    layer = tf.matmul(a,C)
    return layer

def get_NN_layer(x, laters_dimensions, phase_train=None, scope="NNLayer"):
    (D_l_1, D_l) = laters_dimensions
    with tf.name_scope(scope):
        W = tf.Variable(tf.truncated_normal(shape=[D_l_1, D_l], mean=0.0, stddev=1.0), trainable=True)
        b = tf.Variable(tf.constant(1.0, shape=[D_l]), trainable=True)
        z = tf.matmul(x,W) + b
        if phase_train is not None:
            #z = standard_batch_norm(z, 1, phase_train)
            z = batch_norm_layer(z, phase_train, scope_bn=scope+'BN')
        layer = tf.nn.relu(z) # (M x D1) = (M x D) * (D x D1)
    return layer

def get_summation_layer(x, dimensions_list, phase_train=None, scope="SumLayer"):
    #D_in = len(x)
    (D_in, D_out) = (dimensions_list[0], dimensions_list[-1])
    print 'get_summation_layer'
    with tf.name_scope(scope):
        C = tf.Variable( tf.truncated_normal(shape=[D_in,D_out]) )
        a = tf.matmul(x, C)
    return a

def build_NN(x, dimensions_list, phase_train=None):
    current_layer = x
    nb_layers = len(dimensions_list)
    #go through hidden layers
    for l in range(1,nb_layers-1):
        (D_l_1, D_l) = (dimensions_list[l-1], dimensions_list[l])
        current_layer = get_NN_layer(current_layer, (D_l_1, D_l), phase_train)
    f = get_summation_layer(x, phase_train, dimensions_list)
    return f

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

def batch_norm_layer(x,train_phase,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=True,
    reuse=None, # is this right?
    trainable=True,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=False,
    reuse=True, # is this right?
    trainable=True,
    scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

def BatchNorm_my_github_ver(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                lambda: batch_norm(inputT, is_training=True,
                                   center=False, updates_collections=None, scope=scope),
                lambda: batch_norm(inputT, is_training=False,
                                   updates_collections=None, center=False, scope=scope, reuse = True))

def BatchNorm_GitHub_Ver(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                lambda: batch_norm(inputT, is_training=True,
                                   center=False, updates_collections=None, scope=scope),
                lambda: batch_norm(inputT, is_training=False,
                                   updates_collections=None, center=False, scope=scope, reuse = True))
