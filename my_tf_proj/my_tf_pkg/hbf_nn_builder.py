import numpy as np

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def hello_world():
    print "Hello World!"

## builders for Networks

def build_HBF(x, dims, inits, phase_train=None):
    (_,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
    return layer

def build_standard_NN(x, dims, inits, phase_train=None):
    (_,inits_W,inits_b) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_NN_layer(l=str(l), x=layer, init=(inits_W[l],inits_b[l]), dims=(dims[l-1], dims[l]), phase_train=phase_train)
    return layer

## build layers blocks NN

def build_HBF2(x, dims, inits, phase_train=None):
    (_,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_HBF_layer2(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
    return layer

def get_HBF_layer2(l, x, dims, init, phase_train=None, layer_name='HBFLayer'):
    (init_W,init_S) = init
    with tf.name_scope(layer_name+l):
        with tf.name_scope('templates'+l):
            W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
        with tf.name_scope('rbf_stddev'+l):
            print '-->',init_S
            S = tf.get_variable(name='S'+l, dtype=tf.float64, initializer=init_S, regularizer=None, trainable=True)
            beta = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
        with tf.name_scope('Z'+l):
            WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) # (1 x D^(l)) = sum( (D^(l-1) x D^(l)), 0 )
            XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
            # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
            Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
            Z = beta * ( Delta_tilde ) # (M x D^(l))
        if phase_train is not None:
            Z = standard_batch_norm(l, Z , 1, phase_train)
        with tf.name_scope('A'+l):
            A = tf.exp(Z) # (M x D^(l))
    var_prefix = 'vars_'+layer_name+l
    put_summaries(var=W,prefix_name=var_prefix+W.name,suffix_text=W.name)
    put_summaries(var=S,prefix_name=var_prefix+S.name,suffix_text=S.name)
    act_stats = 'acts_'+layer_name+l
    put_summaries(Z,prefix_name=act_stats+'Z'+l,suffix_text='Z'+l)
    put_summaries(A,prefix_name=act_stats+'A'+l,suffix_text='A'+l)
    put_summaries(Delta_tilde,prefix_name=act_stats+'Delta_tilde'+l,suffix_text='Delta_tilde'+l)
    put_summaries(beta,prefix_name=act_stats+'beta'+l,suffix_text='beta'+l)
    return A

def put_summaries(var, prefix_name, suffix_text = ''):
    """Attach a lot of summaries to a Tensor."""
    prefix_title = prefix_name+'/'
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary(prefix_title+'mean'+suffix_text, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary(prefix_title+'stddev'+suffix_text, stddev)
        tf.scalar_summary(prefix_title+'max'+suffix_text, tf.reduce_max(var))
        tf.scalar_summary(prefix_title+'min'+suffix_text, tf.reduce_min(var))
        tf.histogram_summary(prefix_name, var)

# def put_summaries_absolute_val(var, prefix_name, suffix_text = ''):
#     """Attach a lot of summaries to a Tensor to check is absolute value"""
#     prefix_title = prefix_name+'/'
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.scalar_summary(prefix_title+'mean'+suffix_text, mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
#         tf.scalar_summary(prefix_title+'stddev'+suffix_text, stddev)
#         tf.scalar_summary(prefix_title+'max'+suffix_text, tf.reduce_max(var))
#         tf.scalar_summary(prefix_title+'min'+suffix_text, tf.reduce_min(var))
#         tf.histogram_summary(prefix_name, var)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def get_NN_layer(l, x, dims, init, phase_train=None, scope="NNLayer"):
    (init_W,init_b) = init
    with tf.name_scope(scope+l):
        W = get_W(init_W, l, x, dims, init)
        b = tf.get_variable(name='b'+l, dtype=tf.float64, initializer=init_b, regularizer=None, trainable=True)
        with tf.name_scope('Z'+l):
            z = tf.matmul(x,W) + b
            if phase_train is not None:
                #z = standard_batch_norm(l, z, 1, phase_train)
                z = add_batch_norm_layer(l, z, phase_train)
        with tf.name_scope('A'+l):
            a = tf.nn.relu(z) # (M x D1) = (M x D) * (D x D1)
            #a = tf.sigmoid(z)
        with tf.name_scope('sumarries'+l):
            W = tf.histogram_summary('W'+l, W)
            b = tf.histogram_summary('b'+l, b)
    return a

def get_W(init_W, l, x, dims, init):
    (dim_input,dim_out) = dims
    if isinstance(init_W, tf.python.framework.ops.Tensor):
        print 'isinstance'
        W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
    else:
        W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True, shape=[dim_input,dim_out])
    return W

def nn_layer(x, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            W = weight_variable([input_dim, output_dim])
            variable_summaries(W, layer_name + '/weights')
        with tf.name_scope('biases'):
            b = bias_variable([output_dim])
            variable_summaries(b, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            Z = tf.matmul(x, W) + b
            tf.histogram_summary(layer_name + '/pre_activations', Z)
        A = act(Z, 'activation')
        tf.histogram_summary(layer_name + '/activations', A)
        return A

def add_batch_norm_layer(l, x, phase_train, n_out=1, scope='BN'):
    #bn_layer = standard_batch_norm(l, x, n_out, phase_train, scope='BN')
    #bn_layer = batch_norm_layer(x,phase_train,scope_bn=scope+l)
    bn_layer = batch_norm_layer(x,phase_train,scope_bn=scope+l,trainable=True)
    #bn_layer = batch_norm_layer(x,phase_train,scope_bn=scope+l,trainable=False)
    return bn_layer

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

def batch_norm_layer(x,phase_train,scope_bn,trainable=True):
    print '======> official BN'
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None, # is this right?
    trainable=trainable,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True, # is this right?
    trainable=trainable,
    scope=scope_bn)
    z = tf.cond(phase_train, lambda: bn_train, lambda: bn_inference)
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

##

def build_summed_NN(x, dims, inits, phase_train=None):
    (inits_C,inits_W,inits_b) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_NN_layer(str(l),layer,dims,(inits_W[l],inits_b[l]), phase_train=None)
        layer = get_summation_layer(str(l),layer,inits_C[l])
    return layer

def build_summed_HBF(x, dims, inits, phase_train=None):
    (inits_C,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in xrange(1,nb_hidden_layers): # from 1 to L-1
        layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
        layer = get_summation_layer(str(l),layer,inits_C[l])
    return layer

def get_summation_layer(l, x, init, layer_name="SumLayer"):
    with tf.name_scope(layer_name+l):
        #print init
        C = tf.get_variable(name='C', dtype=tf.float64, initializer=init, regularizer=None, trainable=True)
        layer = tf.matmul(x, C)
    var_prefix = 'vars_'+layer_name+l
    put_summaries(C, prefix_name=var_prefix+'C', suffix_text = 'C')
    return layer
