import csv

with open('data.json') as data_file:
    data = json.load(data_file)

## Data sets
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape

## HBF/NN params
dims = [D,16,D_out]
#dims = [D,24,24,D_out]
mu = len(dims)*[0.0]
std = len(dims)*[0.1]
#std = [None,1,1,1]
init_constant = 1
#b_init = len(dims)*[init_constant]
b_init = len(dims)*[init_constant]
#b_init = [None, 1, 1, None]
S_init = b_init
init_type = 'truncated_normal'
#init_type = 'data_init'
#init_type = 'kern_init'
#init_type = 'kpp_init'
model = 'standard_nn'
#model = 'hbf'
#
max_to_keep = 10
#BN
bn = False
if bn:
    phase_train = tf.placeholder(tf.bool, name='phase_train') ##BN ON
else:
    phase_train = None

## Make Model
x = tf.placeholder(tf.float64, shape=[None, D], name='x-input') # M x D
nb_layers = len(dims)-1
nb_hidden_layers = nb_layers-1
print( '-----> Running model: %s. (nb_hidden_layers = %d, nb_layers = %d)' % (model,nb_hidden_layers,nb_layers) )
print( '-----> Units: %s)' % (dims) )
if model == 'standard_nn':
    (inits_C,inits_W,inits_b) = mtf.get_initilizations_standard_NN(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
    with tf.name_scope("standardNN") as scope:
        mdl = mtf.build_standard_NN(x,dims,(inits_C,inits_W,inits_b),phase_train)
        mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
