import numpy as np
import tensorflow as tf

def hello_world():
    print "Hello World!"

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

def get_summation_layer(x, dimensions_list, phase_train=None, scope="SumLayer"):
    (D_in, D_out) = (dimensions_list[0], dimensions_list[-1])
    with tf.name_scope(scope):
        C = tf.Variable( tf.truncated_normal(shape=[D_in,D_out]) )
        a = tf.matmul(x, C)
    return a

## Standardn NN

def get_NN_layer(x, dims, init, phase_train=None, scope="NNLayer"):
    init_W,init_b = init
    with tf.name_scope(scope):
        W = tf.Variable(initsW, trainable=True)
        b = tf.Variable(initsb, trainable=True)
        z = tf.matmul(x,W) + b
        if phase_train is not None:
            z = standard_batch_norm(z, 1, phase_train)
        layer = tf.nn.relu(z) # (M x D1) = (M x D) * (D x D1)
    return layer

def build_NN(x, dims, inits, phase_train=None):
    init_W,init_b = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_NN_layer(x=layer, init=(init_W[l],init_b[l]), dims=(dims[l-1], dims[l]), phase_train)
    return layer

## Final Layer

def get_summation_layer(x, init, scope="SumLayer"):
    with tf.name_scope(scope):
        C = tf.Variable( init )
        a = tf.matmul(x, C)
    return a

## BN

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
