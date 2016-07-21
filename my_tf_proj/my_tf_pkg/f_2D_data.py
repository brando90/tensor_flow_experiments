import numpy as np
import json
from sklearn.cross_validation import train_test_split

# def f2D_task2(X,Y, nb_recursive_layers=2, c1=0.3*np.pi, c2=0.4*np.pi ):
#     # input layer
#     X_1, X_2 = X[:,1], Y[:,2]
#     # first layer
#     A_1 = np.sin( c*np.multiply(X_1, X_2) )
#     A_2 = np.cos( c2(X_1 + X_2) + np.pi/3 )
#     # recursive layer
#     for l in range(nb_recursive_layers):
#         A_1 = np.multiply(3*A_1, A_2)
#         A_2 = c2( np.power(X1,2) + np.power(X2,2)) - 1
#     f = 2*A1 + 3*A2
#     return f
#
# def f2D_task2_draft(X,Y, nb_recursive_layers=2, c1=0.3*np.pi, c2=0.4*np.pi ):
#     # input layer
#     X_1, X_2 = X[:,1], Y[:,2]
#     # first layer
#     A_1 = np.sin( c*np.multiply(X_1, X_2) )
#     A_2 = np.cos( c2(X_1 + X_2) + np.pi/3 )
#     # recursive layer
#     for l in range(nb_recursive_layers):
#         A_1 = np.multiply(3*A_1, A_2)
#         A_2 = c2( np.power(X1,2) + np.power(X2,2)) - 1
#     f = 2*A1 + 3*A2
#     return f

#

def generate_meshgrid_f2D_task2(N=60000,start_val=-1,end_val=1, nb_recursive_layers=2, c1=0.3*np.pi, c2=0.4*np.pi ):
    (X,Y) = generate_meshgrid(N,start_val,end_val)
    (dim_x, dim_y) = X.shape
    Z = np.zeros(X.shape)
    for dx in range(dim_x):
        for dy in range(dim_y):
            x = X[dx, dy]
            y = Y[dx, dy]
            f = f2D_func(x_1=x,x_2=y,nb_recursive_layers=nb_recursive_layers)
            Z[dx, dy] = f
    return X,Y,Z

def f2D_func(x_1,x_2, nb_recursive_layers=2,c1=0.03*np.pi,c2=0.04*np.pi,c3=np.pi/3):
    # first layer
    A_1 = np.sin( c1*np.multiply(x_1,x_2) )
    A_2 = np.cos( c2*(x_1+x_2) + c3 )
    # recursive layer
    for l in range(nb_recursive_layers):
        A_1 = np.multiply(3*A_1, A_2)
        A_2 = c2*( np.power(A_1,2) + np.power(A_2,2)) - 1
    f = 2*A_1 + 3*A_2
    return f

##

def generate_meshgrid_h_gabor(N=60000,start_val=-1,end_val=1):
    (X,Y) = generate_meshgrid(N,start_val,end_val)
    # h_gabor
    Z = np.multiply( np.exp( -(np.power(X,2)+np.power(Y,2) )) , np.cos(2*np.pi*(X+Y) ) )
    return X,Y,Z

def generate_meshgrid_h_add(N=60000,start_val=-1,end_val=1):
    (X,Y) = generate_meshgrid(N,start_val,end_val)
    #Z = sin(2*pi*X) + 4*(Y - 0.5).^2; %% h_add
    Z = np.sin(2*np.pi*X) + 4*np.power(Y - 0.5, 2) # h_add
    return X,Y,Z

##

def generate_meshgrid(N,start_val,end_val):
    sqrtN = int(np.ceil(N**0.5)) #N = sqrtN*sqrtN
    N = sqrtN*sqrtN
    x_range = np.linspace(start_val, end_val, sqrtN)
    y_range = np.linspace(start_val, end_val, sqrtN)
    ## make meshgrid
    (X,Y) = np.meshgrid(x_range, y_range)
    return X,Y

def make_mesh_grid_to_data_set(X, Y, Z):
    '''
        want to make data set as:
        ( x = [x1, x2], z = f(x,y) )
        X = [N, D], Z = [Dout, N] = [1, N]
    '''
    (dim_x, dim_y) = X.shape
    N = dim_x * dim_y
    X_data = np.zeros((N,2))
    Y_data = np.zeros((N,1))
    i = 0
    for dx in range(dim_x):
        for dy in range(dim_y):
            # input val
            x = X[dx, dy]
            y = Y[dx, dy]
            x_data = np.array([x, y])
            # func val
            z = Z[dx, dy]
            y_data = z
            # load data set
            X_data[i,:] = x_data
            Y_data[i,:] = y_data
            i=i+1;
    return X_data, Y_data

def make_meshgrid_data_from_training_data(X_data, Y_data):
    N, _ = X_data.shape
    sqrtN = int(np.ceil(N**0.5))
    dim_y = sqrtN
    dim_x = dim_y
    shape = (sqrtN,sqrtN)
    X = np.zeros(shape)
    Y = np.zeros(shape)
    Z = np.zeros(shape)
    i = 0
    for dx in range(dim_x):
        for dy in range(dim_y):
            #x_vec = X_data[:,i]
            #x,y = x_vec(1),x_vec(2)
            x,y = X_data[i,:]
            #x = x_vec(1);
            #y = x_vec(2);
            z = Y_data[i,:]
            X[dx,dy] = x
            Y[dx,dy] = y
            Z[dx,dy] = z
            i = i+1;
    return X,Y,Z

##

# def random_gen_h_add(N=60000,start_val=-1,end_val=1):
#     sqrtN = int(np.ceil(N**0.5)) #N = sqrtN*sqrtN
#     N = sqrtN*sqrtN
#     x_range = np.random.uniform(low=start_val, high=end_val, size=sqrtN)
#     y_range = np.random.uniform(low=start_val, high=end_val, size=sqrtN)
#     (X,Y) = np.meshgrid(x_range, y_range)
#     Z = np.sin(2*np.pi*X) + 4*np.power(Y - 0.5, 2) # h_add
#     return X,Y,Z

##

def generate_data_task2():
    # train
    N_train = 70000
    nb_recursive_layers = 2
    X_surf_train,Y_surf_train,Z_surf_train = mtf.generate_meshgrid_f2D_task2(N=N_train,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
    X_train_whole, Y_train_whole = mtf.make_mesh_grid_to_data_set(X_surf_train, Y_surf_train, Z_surf_train)
    N, K = 60000, N_train
    replace = False
    indices=np.random.choice(a=N_train,size=K,replace=replace) # choose numbers from 0 to D^(1)
    X_train, Y_train = X_train[indices,:] Y_train[indices,:]# M_sub x D
    # cv
    N_cv = 60100
    X_surf_cv,Y_surf_cv,Z_surf_cv = mtf.generate_meshgrid_f2D_task2(N=N_cv,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
    X_cv, Y_cv = mtf.make_mesh_grid_to_data_set(X_surf_cv, Y_surf_cv, Z_surf_cv)
    # test
    N_test = 60150
    X_surf_test,Y_surf_test,Z_surf_test = mtf.generate_meshgrid_f2D_task2(N=N_test,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
    X_test, Y_test = mtf.make_mesh_grid_to_data_set(X_surf_test, Y_surf_test, Z_surf_test)
    return (X_train_whole, Y_train_whole), (X_train, Y_train, X_cv, Y_cv, X_test, Y_test), (X_surf_train,Y_surf_train,Z_surf_train, X_surf_cv,Y_surf_cv,Z_surf_cv, X_surf_test,Y_surf_test,Z_surf_test)

def save_data_task2():
    (X_train_whole, Y_train_whole), (X_train, Y_train, X_cv, Y_cv, X_test, Y_test), (X_surf_train,Y_surf_train,Z_surf_train, X_surf_cv,Y_surf_cv,Z_surf_cv, X_surf_test,Y_surf_test,Z_surf_test) = generate_data_task2()
    file_name = 'f_2d_task2_data_and_more' #.npz will be appended
    numpy.savez(file_name, X_train_whole, Y_train_whole, X_train, Y_train, X_cv, Y_cv, X_test, Y_test, X_surf_train,Y_surf_train,Z_surf_train, X_surf_cv,Y_surf_cv,Z_surf_cv, X_surf_test,Y_surf_test,Z_surf_test)
