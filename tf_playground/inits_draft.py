#import winsound

# def get_initilizations(**kwargs):
#     if kwargs['init_type'] == 'truncated_normal':
#         dims = kwargs['dims']
#         inits_W = [None]
#         for l in range(1,):
#             inits_W.append(tf.truncated_normal(shape=dims[l-1], dims[l], mean=kwargs['mean'], stddev=kwargs['stddev']))
#             init_b.append(tf.constant(0.1, shape=[dims[l]]))
#     elif kwargs['init_type']  == 'data_init':
#         pass
#     return inits_W,init_b

def get_initilizations(init_args):
    if init_args.init_type == 'truncated_normal':
        inits_W = [None]
        inits_b = [None]
        nb_hidden_layers = len(dims)-1
        for l in range(1,nb_hidden_layers):
            inits_W.append( tf.truncated_normal(shape=[init_args.dims[l-1],init_args.dims[l]], mean=init_args.mu[l], stddev=init_args.std[l], dtype=tf.float64) )
            inits_b.append( tf.constant(init_args.b_init[l], shape=[dims[l]], dtype=tf.float64 ) )
        l = len(init_args.dims)-1
        inits_C = [ tf.truncated_normal(dtype=tf.float64, shape=[init_args.dims[l-1],init_args.dims[l]], mean=init_args.mu, stddev=init_args.std) ]
    elif init_args.init_type  == 'data_init':
        X_train = init_args.X_train
        pass
    return (inits_C,inits_W,inits_b)
