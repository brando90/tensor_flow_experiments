import numpy as np
import tensorflow as tf
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
import my_tf_pkg as mtf

import sklearn.cluster.k_means_
#from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import check_random_state

def get_initilizations_standard_NN(init_type,dims,mu,std,b_init,S_init,X_train,Y_train,train_S_type='multiple_S'):
#def get_initilizations_standard_NN(args):
    if  init_type=='truncated_normal':
        inits_W=[None]
        inits_b=[None]
        nb_hidden_layers=len(dims)-1
        for l in range(1,nb_hidden_layers):
            #inits_W.append( tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_W.append( tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_b.append( tf.constant( b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        l=len(dims)-1
        print [dims[l-1],dims[l]]
        inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu, stddev=std, dtype=tf.float64) ]
    elif init_type=='data_init':
        X_train=X_train
        pass
    elif init_type=='xavier':
        inits_W=[None]
        inits_b=[None]
        nb_hidden_layers=len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.contrib.layers.xavier_initializer(dtype=tf.float64) )
            inits_b.append( tf.constant( b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        l=len(dims)-1
        print [dims[l-1],dims[l]]
        inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu, stddev=std, dtype=tf.float64) ]
    return (inits_C,inits_W,inits_b)

def get_initilizations_HBF(init_type,dims,mu,std,b_init,S_init,X_train,Y_train,train_S_type='multiple_S'):
#def get_initilizations_HBF(args):
    print 'train_S_type: ', train_S_type
    nb_hidden_layers=len(dims)-1
    print init_type
    if init_type=='truncated_normal':
        inits_W=[None]
        inits_S=[None]
        inits_C=[None]
        nb_hidden_layers=len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_S.append( get_single_multiple_S(l,S_init,dims,train_S_type) )
            #inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            #inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
        l=len(dims)-1
        inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) ]
    elif init_type=='data_init':
        nb_hidden_layers=len(dims)-1
        inits_W=[None]
        inits_S=[None]

        (subsampled_data_points,W,W_tf)= get_centers_from_data(X_train,dims)
        inits_W.append( W_tf )
        for l in range(1,nb_hidden_layers):
            #inits_S.append( tf.constant( S_init[l], shape=[], dtype=tf.float64 ) )
            #inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            inits_S.append( get_single_multiple_S(l,S_init,dims,train_S_type) )
        l=len(dims)-1
        inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu, stddev=std, dtype=tf.float64) ]
    elif init_type=='kern_init':
        inits_W=[None]
        inits_S=[None]

        (subsampled_data_points,W,W_tf)= get_centers_from_data(X_train,dims)
        inits_W.append( W_tf )

        for l in range(1,nb_hidden_layers):
            inits_S.append( get_single_multiple_S(l,S_init,dims,train_S_type) )
        stddev = S_init[1]
        beta = np.power(1.0/stddev,2)
        Kern = np.exp(-beta*euclidean_distances(X=X_train,Y=subsampled_data_points,squared=True))
        (C,_,_,_) = np.linalg.lstsq(Kern,Y_train)
        inits_C=[tf.constant(C)]
        print report_RBF_error(Kern, C, Y_train)
    elif init_type=='kpp_init':
        inits_W=[None]
        inits_S=[None]

        (centers,W,W_tf) = get_kpp_init(X=X_train,n_clusters=dims[1],random_state=None)
        inits_W.append( W_tf )

        for l in range(1,nb_hidden_layers):
            inits_S.append( get_single_multiple_S(l,S_init,dims,train_S_type) )
            #inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        stddev = S_init[1]
        beta = np.power(1.0/stddev,2)
        Kern = np.exp(-beta*euclidean_distances(X=X_train,Y=centers,squared=True))
        (C,_,_,_) = np.linalg.lstsq(Kern,Y_train)
        inits_C=[tf.constant(C)]
        print report_RBF_error(Kern, C, Y_train)
    elif init_type=='kpp_trun_norm_lq':
        inits_W=[None]
        inits_S=[None]

        (centers,W,W_tf) = get_kpp_init(X=X_train,n_clusters=dims[1],random_state=None)
        inits_W.append( W_tf )

        for l in range(1,nb_hidden_layers):
            inits_S.append( get_single_multiple_S(l,S_init,dims,train_S_type) )
            #inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        stddev = S_init[1]
        beta = np.power(1.0/stddev,2)
        Kern = np.exp(-beta*euclidean_distances(X=X_train,Y=centers,squared=True))
        (C,_,_,_) = np.linalg.lstsq(Kern,Y_train)
        inits_C=[tf.constant(C)]
    elif init_type=='data_trunc_norm_kern':
        inits_W=[None]
        inits_S=[None]

        (subsampled_data_points,W,W_tf)= get_centers_from_data(X_train,dims)
        inits_W.append( W_tf )
        # nb_hidden_layers=len(dims)-1
        for l in xrange(1, nb_hidden_layers):
            #inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            inits_S.append( get_single_multiple_S(l,S_init,dims,train_S_type) )
        for l in xrange(2,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
        stddev = S_init[1]
        beta = np.power(1.0/stddev,2)
        Kern = np.exp(-beta*euclidean_distances(X=X_train,Y=subsampled_data_points,squared=True))
        (C,_,_,_) = np.linalg.lstsq(Kern,Y_train)
        inits_C=[tf.constant(C)]
        #inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) ]
        print report_RBF_error(Kern, C, Y_train)
    print 'DONE INITILIZING'
    return (inits_C,inits_W,inits_S)

def get_single_multiple_S(l,S_init,dims,train_S_type='multiple_S'):
    if train_S_type == 'multiple_S':
        S = tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 )
    elif train_S_type == 'single_S':
        S = tf.constant( S_init[l], shape=[], dtype=tf.float64 )
    return S

def get_centers_from_data(X_train,dims):
    N = X_train.shape[0]
    indices=np.random.choice( N,size=dims[1] )
    subsampled_data_points=X_train[indices,:] # D^(1) x D
    W =  np.transpose( subsampled_data_points )  # D x D^(1)
    W_tf = tf.constant(W)
    return subsampled_data_points,W,W_tf

def get_kpp_init(X,n_clusters,random_state=None):
    random_state = None
    random_state = check_random_state(random_state)
    x_squared_norms = row_norms(X, squared=True)
    centers = sklearn.cluster.k_means_._k_init(X, n_clusters, random_state=random_state,x_squared_norms=x_squared_norms) # n_clusters x D
    W =  np.transpose( centers )  # D x D^(1)
    W_tf = tf.constant(W)
    return centers,W,W_tf

def report_RBF_error(Kern, C, Y):
    Y_pred = np.dot( Kern , C )
    error = sklearn.metrics.mean_squared_error(Y, Y_pred)
    return error

# def get_initilizations_summed_NN(init_type,dims,mu,std,b_init,S_init,X_train,Y_train):
# #def get_initilizations_summed_NN(args):
#     if  init_type=='truncated_normal':
#         inits_W=[None]
#         inits_b=[None]
#         inits_C=[None]
#         nb_hidden_layers=len(dims)-1
#         for l in range(1,nb_hidden_layers):
#             inits_W.append( tf.truncated_normal(shape=[1,dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
#             inits_b.append( tf.constant( b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
#             inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
#     elif  init_type=='data_init':
#         X_train=X_train
#         pass
#     return (inits_C,inits_W,inits_b)
#
# def get_initilizations_summed_HBF(init_type,dims,mu,std,b_init,S_init,X_train,Y_train):
# #def get_initilizations_summed_HBF(args):
#     if  init_type=='truncated_normal':
#         inits_W=[None]
#         inits_S=[None]
#         inits_C=[None]
#         nb_hidden_layers=len(dims)-1
#         for l in range(1,nb_hidden_layers):
#             inits_W.append( tf.truncated_normal(shape=[1,dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
#             inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
#             inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
#     elif  init_type=='data_init':
#         X_train=X_train
#         pass
#     return (inits_C,inits_W,inits_S)
