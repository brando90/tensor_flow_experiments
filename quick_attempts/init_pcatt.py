import tensorflow as tf

import namespaces as ns
# import collections
#
# Init = collections.namedtuple('Init', ['bias', 'weight'])
# init1 = Init(bias=f1, weights=f2)
# init2 = Init(bias=f1, weights=f2)

init = ns.FrozenNamespace(bias=init1, weights=init2)
#init = ns.Namespace(weights=init2, bias=init1)
init.bias(...)
init.weights(...)


#def get_nn(x,dimensions,inits, X_train=None):
def get_nn(x, nn_args):
  layer = x
  for l, init in enumerate(nn_args.inits):
    #ReLu(Wx+b)
    #(D_l_1, D_l) = (dimensions[l-1],dimensions[l])
    dim_prev, dim = nn_args.dimensions[l - 1:l + 1]
    init_weights, init_bias = init

    W = tf.Variable(init_weights([dim_prev, dim], nn_args.X_train))
    b = tf.Variable(init_bias([dim], nn_args.X_train))
    layer = tf.nn_relu(tf.matmul(layer, W) + b)
  return layer

def init(shape, X_train=None):
  if X_train:
    # TODO implement this
    pass
  return tf.truncated_normal(shape=shape)

(N,D) = X_train.shape
D_out = Y_train.shape[1]
nn_args = ns.Namespace(
  dimensions=[D] + [10]*(length -2) + [D_out],
  X_train=X_train
)
length = 5
x = tf.placeholder(X_train)
y = get_nn(x, nn_args)
