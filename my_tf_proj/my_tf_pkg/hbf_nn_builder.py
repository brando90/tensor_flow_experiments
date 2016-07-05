import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import sklearn.cluster.k_means_
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import check_random_state

def hello_world():
    print "Hello World!"

## builders for Networks

def build_summed_HBF(x, dims, inits, phase_train=None):
    (inits_C,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
        layer = get_summation_layer(str(l),layer,inits_C[l])
    return layer

def build_HBF(x, dims, inits, phase_train=None):
    (_,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
    return layer

def build_summed_NN(x, dims, inits, phase_train=None):
    (inits_C,inits_W,inits_b) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_NN_layer(str(l),layer,dims,(inits_W[l],inits_b[l]), phase_train=None)
        layer = get_summation_layer(str(l),layer,inits_C[l])
    return layer

def build_standard_NN(x, dims, inits, phase_train=None):
    (_,inits_W,inits_b) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_NN_layer(l=str(l), x=layer, init=(inits_W[l],inits_b[l]), dims=(dims[l-1], dims[l]), phase_train=phase_train)
    return layer

## build layers blocks NN

def get_summation_layer(l, x, init, scope="SumLayer"):
    with tf.name_scope(scope+l):
        #print init
        C = tf.get_variable(name='C'+l, dtype=tf.float64, initializer=init, regularizer=None, trainable=True)
        layer = tf.matmul(x, C)
    C = tf.histogram_summary('C'+l, C)
    return layer

def get_HBF_layer(l, x, dims, init, phase_train=None, scope="HBFLayer"):
    (init_W,init_S) = init
    #i = int(l)
    with tf.name_scope(scope+l):
        with tf.name_scope('Z'+l):
            W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
            S = tf.get_variable(name='S'+l, dtype=tf.float64, initializer=init_S, regularizer=None, trainable=True)
            # if phase_train is not None:
            #     x = standard_batch_norm(l+'_', x , 1,phase_train)
            WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
            XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
            # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
            Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
            # if phase_train is not None:
            #     Delta_tilde = standard_batch_norm(l, Delta_tilde, 1,phase_train)
            beta = 0.5*tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
            Z = beta * ( Delta_tilde ) # (M x D^(l))
        if phase_train is not None:
            #Z = standard_batch_norm(l, Zd , 1,phase_train)
            Z = standard_batch_norm(l, Z , 1, phase_train)
        with tf.name_scope('A'+l):
            layer = tf.exp(Z) # (M x D^(l))
    W = tf.histogram_summary('W'+l, W)
    b = tf.histogram_summary('S'+l, S)
    return layer

def get_NN_layer(l, x, dims, init, phase_train=None, scope="NNLayer"):
    (init_W,init_b) = init
    with tf.name_scope(scope+l):
        W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
        b = tf.get_variable(name='b'+l, dtype=tf.float64, initializer=init_b, regularizer=None, trainable=True)
        z = tf.matmul(x,W) + b
        if phase_train is not None:
            z = standard_batch_norm(l, z, 1, phase_train)
        layer = tf.nn.relu(z) # (M x D1) = (M x D) * (D x D1)
    W = tf.histogram_summary('W'+l, W)
    b = tf.histogram_summary('b'+l, b)
    return layer

def standard_batch_norm(l, x, n_out, phase_train, scope='BN'):
    """
    Batch normalization on feedforward maps.
    Args:
        x:           Vector
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope+l):
        #beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float64 ), name='beta', trainable=True, dtype=tf.float64 )
        #gamma = tf.Variable(tf.constant(1.0, shape=[n_out],dtype=tf.float64 ), name='gamma', trainable=True, dtype=tf.float64 )
        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float64)
        init_gamma = tf.constant(1.0, shape=[n_out],dtype=tf.float64)
        beta = tf.get_variable(name='beta'+l, dtype=tf.float64, initializer=init_beta, regularizer=None, trainable=True)
        gamma = tf.get_variable(name='gamma'+l, dtype=tf.float64, initializer=init_gamma, regularizer=None, trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
