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
# def generate_data_task2():
#     print 'hello'
#     # train
#     N_train = 70000
#     nb_recursive_layers = 2
#     X_surf_train,Y_surf_train,Z_surf_train = generate_meshgrid_f2D_task2(N=N_train,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
#     X_train_whole, Y_train_whole = make_mesh_grid_to_data_set(X_surf_train, Y_surf_train, Z_surf_train)
#     K = 60000
#     replace = False
#     indices=np.random.choice(a=N_train,size=K,replace=replace) # choose numbers from 0 to D^(1)
#     X_train, Y_train = X_train_whole[indices,:], Y_train_whole[indices,:]# M_sub x D
#     # cv
#     N_cv = 60100
#     X_surf_cv,Y_surf_cv,Z_surf_cv = generate_meshgrid_f2D_task2(N=N_cv,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
#     X_cv, Y_cv = make_mesh_grid_to_data_set(X_surf_cv, Y_surf_cv, Z_surf_cv)
#     # test
#     N_test = 60150
#     X_surf_test,Y_surf_test,Z_surf_test = generate_meshgrid_f2D_task2(N=N_test,start_val=-10,end_val=10, nb_recursive_layers=nb_recursive_layers)
#     X_test, Y_test = make_mesh_grid_to_data_set(X_surf_test, Y_surf_test, Z_surf_test)
#     return (X_train_whole, Y_train_whole), (X_train, Y_train, X_cv, Y_cv, X_test, Y_test), (X_surf_train,Y_surf_train,Z_surf_train, X_surf_cv,Y_surf_cv,Z_surf_cv, X_surf_test,Y_surf_test,Z_surf_test)

#

def f2D_func_task2_2(x_1,x_2,nb_recursive_layers=2):
    #first layer
    #c_h = 0.2*np.pi
    #c_h = 0.01*np.pi
    c_h = 1.2*np.pi
    #h = np.cos(c_h*( x_1+x_2))
    #h = np.cos(c_h*( x_1*x_2))
    h = np.cos(c_h*( x_1*x_2**2))
    # recursive layer
    c_a1 = 1.2
    A = c_a1*h
    for l in range(nb_recursive_layers):
        #A = (1.01*A**2+1.01*A+1.1)**.5
        #A = A+1.1*A**2+(A**2+0.5*A**4)**.5
        #A = A*0.5*(1.01*A**2+1.01*A + 1.1)**0.5
        #A = A*np.log(A**2)
        #A = (np.abs(A) )**1.1
        #A = (A**2+0.5*A**4)**0.5
        #A = 1.01*np.sin(0.1*np.pi*A)*(1.01*A**2+1.01*A + 1.1)**0.4
        #A = A*np.sin(1.0/A)/2
        #A = A*np.sin( 0.5*np.pi*np.abs(A)**(-0.37) )/2
        A = 0.99*A*np.sin( 0.1*np.pi*np.log(np.abs(A)) )
    f = 60*A
    return f

def f2D_func_task2_3(x_1,x_2,nb_recursive_layers=2):
    #first layer
    c_h1 = 0.5*np.pi
    c_h2 = 1
    xx3 = x_1*x_2
    h_1, h_2 = np.sin(c_h1*(x_1 + x_2)), np.cos(c_h2*(xx3))
    # recursive layer
    c_a1, c_a2 = 1.5, 0.9
    A = c_a1*h_1 + c_a2*h_2
    for l in range(nb_recursive_layers):
        A = 0.5*(1*A**2+1.01*A + 1.1)**0.6
    return A

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

## normal meshgrid

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

def generate_meshgrid_f2D_task2_2(N=60000,start_val=-1,end_val=1, nb_recursive_layers=2):
    (X,Y) = generate_meshgrid(N,start_val,end_val)
    (dim_x, dim_y) = X.shape
    Z = np.zeros(X.shape)
    for dx in range(dim_x):
        for dy in range(dim_y):
            x = X[dx, dy]
            y = Y[dx, dy]
            f = f2D_func_task2_2(x_1=x,x_2=y,nb_recursive_layers=nb_recursive_layers)
            Z[dx, dy] = f
    return X,Y,Z

def generate_meshgrid_f2D_task2_func(func,N=60000,start_val=-1,end_val=1,nb_recursive_layers=2):
    (X,Y) = generate_meshgrid(N,start_val,end_val)
    (dim_x, dim_y) = X.shape
    Z = np.zeros(X.shape)
    for dx in range(dim_x):
        for dy in range(dim_y):
            x = X[dx, dy]
            y = Y[dx, dy]
            f = func(x_1=x,x_2=y,nb_recursive_layers=nb_recursive_layers)
            Z[dx, dy] = f
    return X,Y,Z

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

## random mesh grid

def generate_random_meshgrid_h_gabor(N=60000,start_val=-1,end_val=1):
    (X,Y) = generate_random_meshgrid(N,start_val,end_val)
    # h_gabor
    Z = np.multiply( np.exp( -(np.power(X,2)+np.power(Y,2) )) , np.cos(2*np.pi*(X+Y) ) )
    return X,Y,Z

def random_random_gen_h_add(N=60000,start_val=-1,end_val=1):
    (X,Y) = generate_random_meshgrid(N,start_val,end_val)
    #Z = sin(2*pi*X) + 4*(Y - 0.5).^2; %% h_add
    Z = np.sin(2*np.pi*X) + 4*np.power(Y - 0.5, 2) # h_add
    return X,Y,Z

def generate_random_meshgrid_f2D_task2(N=60000,start_val=-1,end_val=1, nb_recursive_layers=2, c1=0.3*np.pi, c2=0.4*np.pi ):
    (X,Y) = generate_random_meshgrid(N,start_val,end_val)
    (dim_x, dim_y) = X.shape
    Z = np.zeros(X.shape)
    for dx in range(dim_x):
        for dy in range(dim_y):
            x = X[dx, dy]
            y = Y[dx, dy]
            f = f2D_func(x_1=x,x_2=y,nb_recursive_layers=nb_recursive_layers)
            Z[dx, dy] = f
    return X,Y,Z

def generate_random_meshgrid_f2D_func(func,nb_recursive_layers, N=60000,start_val=-1,end_val=1, c1=0.3*np.pi, c2=0.4*np.pi ):
    (X,Y) = generate_random_meshgrid(N,start_val,end_val)
    (dim_x, dim_y) = X.shape
    Z = np.zeros(X.shape)
    for dx in range(dim_x):
        for dy in range(dim_y):
            x = X[dx, dy]
            y = Y[dx, dy]
            f = func(x_1=x,x_2=y,nb_recursive_layers=nb_recursive_layers)
            Z[dx, dy] = f
    return X,Y,Z

## helper functions

def generate_random_meshgrid(N,start_val,end_val):
    sqrtN = int(np.ceil(N**0.5)) #N = sqrtN*sqrtN
    N = sqrtN*sqrtN
    x_range = np.sort( np.random.uniform(low=start_val, high=end_val, size=sqrtN) )
    y_range = np.sort( np.random.uniform(low=start_val, high=end_val, size=sqrtN) )
    ## make meshgrid
    (X,Y) = np.meshgrid(x_range, y_range)
    return X,Y

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

# def generate_data_task2(N_train=60000,N_cv=60000,N_test=60000, nb_recursive_layers=2, start_val=-10,end_val=-10):
#     # train
#     X_mesh_train, Y_mesh_train, Z_mesh_train = generate_random_meshgrid_f2D_task2(N=N_train,start_val=-start_val,end_val=end_val,nb_recursive_layers=nb_recursive_layers)
#     X_train, Y_train = make_mesh_grid_to_data_set(X_mesh_train, Y_mesh_train, Z_mesh_train)
#     # cv
#     X_mesh_cv, Y_mesh_cv, Z_mesh_cv = generate_random_meshgrid_f2D_task2(N=N_cv,start_val=-start_val,end_val=end_val,nb_recursive_layers=nb_recursive_layers)
#     X_cv, Y_cv = make_mesh_grid_to_data_set(X_mesh_cv, Y_mesh_cv, Z_mesh_cv)
#     # test
#     X_mesh_test, Y_mesh_test, Z_mesh_test = generate_random_meshgrid_f2D_task2(N=N_test,start_val=-start_val,end_val=end_val,nb_recursive_layers=nb_recursive_layers)
#     X_test, Y_test = make_mesh_grid_to_data_set(X_mesh_test, Y_mesh_test, Z_mesh_test)
#     return (X_train,Y_train, X_cv,Y_cv, X_test,Y_test), (X_mesh_train,Y_mesh_train,Z_mesh_train, X_mesh_cv,Y_mesh_cv,Z_mesh_cv, X_mesh_test,Y_mesh_test,Z_mesh_test)

def generate_data_task2_func(func,nb_recursive_layers, N_train=60000,N_cv=60000,N_test=60000, start_val=-1,end_val=-1):
    # train
    X_mesh_train, Y_mesh_train, Z_mesh_train = generate_random_meshgrid_f2D_func(func=func,N=N_train,start_val=-start_val,end_val=end_val,nb_recursive_layers=nb_recursive_layers)
    X_train, Y_train = make_mesh_grid_to_data_set(X_mesh_train, Y_mesh_train, Z_mesh_train)
    # cv
    X_mesh_cv, Y_mesh_cv, Z_mesh_cv = generate_random_meshgrid_f2D_func(func=func,N=N_cv,start_val=-start_val,end_val=end_val,nb_recursive_layers=nb_recursive_layers)
    X_cv, Y_cv = make_mesh_grid_to_data_set(X_mesh_cv, Y_mesh_cv, Z_mesh_cv)
    # test
    X_mesh_test, Y_mesh_test, Z_mesh_test = generate_random_meshgrid_f2D_func(func=func,N=N_test,start_val=-start_val,end_val=end_val,nb_recursive_layers=nb_recursive_layers)
    X_test, Y_test = make_mesh_grid_to_data_set(X_mesh_test, Y_mesh_test, Z_mesh_test)
    return (X_train,Y_train, X_cv,Y_cv, X_test,Y_test), (X_mesh_train,Y_mesh_train,Z_mesh_train, X_mesh_cv,Y_mesh_cv,Z_mesh_cv, X_mesh_test,Y_mesh_test,Z_mesh_test)

# def save_data_task2():
#     print 'save_data_task2'
#     (X_train,Y_train, X_cv,Y_cv, X_test,Y_test), (X_mesh_train,Y_mesh_train,Z_mesh_train, X_mesh_cv,Y_mesh_cv,Z_mesh_cv, X_mesh_test,Y_mesh_test,Z_mesh_test) = generate_data_task2()
#     file_name = 'f_2d_task2_ml_data_and_mesh' #.npz will be appended
#     print 'file_name: ', file_name
#     np.savez(file_name, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test, X_mesh_train=X_mesh_train,Y_mesh_train=Y_mesh_train,Z_mesh_train=Z_mesh_train, X_mesh_cv=X_mesh_cv,Y_mesh_cv=Y_mesh_cv,Z_mesh_cv=Z_mesh_cv, X_mesh_test=X_mesh_test,Y_mesh_test=Y_mesh_test,Z_mesh_test=Z_mesh_test)

def save_data_task2_func(func,file_name,nb_recursive_layers):
    print 'save_data_task2'
    (X_train,Y_train, X_cv,Y_cv, X_test,Y_test), (X_mesh_train,Y_mesh_train,Z_mesh_train, X_mesh_cv,Y_mesh_cv,Z_mesh_cv, X_mesh_test,Y_mesh_test,Z_mesh_test) = generate_data_task2_func(func=func,nb_recursive_layers=nb_recursive_layers)
    print 'file_name: ', file_name
    np.savez(file_name, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test, X_mesh_train=X_mesh_train,Y_mesh_train=Y_mesh_train,Z_mesh_train=Z_mesh_train, X_mesh_cv=X_mesh_cv,Y_mesh_cv=Y_mesh_cv,Z_mesh_cv=Z_mesh_cv, X_mesh_test=X_mesh_test,Y_mesh_test=Y_mesh_test,Z_mesh_test=Z_mesh_test)
