import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances
import my_tf_pkg as mtf

# def get_Z_tf(x,W,S,l='layer'):
#     W = tf.Variable(W, name='W'+l, trainable=True, dtype=tf.float64)
#     S = tf.Variable(S, name='S'+l, trainable=True, dtype=tf.float64)
#     WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
#     XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
#     # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
#     Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX)
#     return S*Delta_tilde
#
# def get_z_np(x,W,S):
#     WW = np.sum(np.multiply(W,W), axis=0, dtype=None, keepdims=True)
#     XX = np.sum(np.multiply(x,x), axis=1, dtype=None, keepdims=True)
#     Delta_tilde = 2.0*np.dot(x,W) - (WW + XX)
#     return S*Delta_tilde

W = np.random.rand(4,3) # D x M
x = np.random.rand(5,4) # M' x D
S = 0.9

#x_tf = tf.constant(x)
#sklearn.metrics.pairwise.euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None)[source]

with tf.Session() as sess:
    x_const = tf.constant(x,dtype=tf.float64)
    beta = mtf.get_beta_tf(S)
    Z_tf = beta * mtf.get_Z_tf(x_const,W)
    sess.run(tf.initialize_all_variables())
    Z_tf = sess.run(Z_tf)

    beta = mtf.get_beta_np(S)
    Z_np = beta*mtf.get_z_np(x,W)

    Z_sk = -beta*euclidean_distances(X=x,Y=np.transpose(W),squared=True)


    print 'tf'
    print Z_tf
    print 'np'
    print Z_np
    print 'sk'
    print Z_sk
