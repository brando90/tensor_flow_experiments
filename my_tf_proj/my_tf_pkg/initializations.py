import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances
import my_tf_pkg as mtf

def get_initilizations_standard_NN(init_type,dims,mu,std,b_init,S_init,X_train,Y_train):
#def get_initilizations_standard_NN(args):
    if  init_type=='truncated_normal':
        inits_W=[None]
        inits_b=[None]
        nb_hidden_layers=len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_b.append( tf.constant( b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        l=len(dims)-1
        inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu, stddev=std, dtype=tf.float64) ]
    elif  init_type=='data_init':
        X_train=X_train
        pass
    return (inits_C,inits_W,inits_b)

def get_initilizations_summed_NN(init_type,dims,mu,std,b_init,S_init,X_train,Y_train):
#def get_initilizations_summed_NN(args):
    if  init_type=='truncated_normal':
        inits_W=[None]
        inits_b=[None]
        inits_C=[None]
        nb_hidden_layers=len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[1,dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_b.append( tf.constant( b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
    elif  init_type=='data_init':
        X_train=X_train
        pass
    return (inits_C,inits_W,inits_b)

def get_initilizations_summed_HBF(init_type,dims,mu,std,b_init,S_init,X_train,Y_train):
#def get_initilizations_summed_HBF(args):
    if  init_type=='truncated_normal':
        inits_W=[None]
        inits_S=[None]
        inits_C=[None]
        nb_hidden_layers=len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[1,dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
    elif  init_type=='data_init':
        X_train=X_train
        pass
    return (inits_C,inits_W,inits_S)

def get_initilizations_HBF(init_type,dims,mu,std,b_init,S_init,X_train,Y_train):
#def get_initilizations_HBF(args):
    nb_hidden_layers=len(dims)-1
    if init_type=='truncated_normal':
        inits_W=[None]
        inits_S=[None]
        inits_C=[None]
        nb_hidden_layers=len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[1,dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            #inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
        l=len(dims)-1
        inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu, stddev=std, dtype=tf.float64) ]
    elif init_type=='data_init':
        nb_hidden_layers=len(dims)-1
        inits_W=[None]
        inits_S=[None]
        indices=np.random.choice( X_train.shape[0],size=dims[1])
        #print np.transpose( X_train[indices,:]).shape
        inits_W.append(np.transpose( X_train[indices,:]) )
        for l in range(1,nb_hidden_layers):
            #inits_W.append( tf.truncated_normal(shape=[1,dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            #inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
        l=len(dims)-1
        inits_C=[ tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu, stddev=std, dtype=tf.float64) ]
    elif init_type=='kern_init':
        inits_W=[None]
        inits_S=[None]
        indices=np.random.choice(a=X_train.shape[0],size=dims[1]) # choose numbers from 0 to D^(1)
        subsampled_data_points=X_train[indices,:] # M_sub x D
        inits_W.append( np.transpose(subsampled_data_points) ) # D x M_sub
        for l in range(1,nb_hidden_layers):
            #inits_W.append( tf.truncated_normal(shape=[1,dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64) )
            inits_S.append( tf.constant( S_init[l], shape=[dims[l]], dtype=tf.float64 ) )
            #inits_C.append( tf.truncated_normal(shape=[dims[l],1], mean=mu[l], stddev=std[l], dtype=tf.float64) )
        K=mtf.get_kernel_matrix( X_train,subsampled_data_points, S_init[1])
        (C,_,_,_)=np.linalg.lstsq(K, Y_train)
        inits_C=[C]
    return (inits_C,inits_W,inits_S)

#def choose_random_data_points():
