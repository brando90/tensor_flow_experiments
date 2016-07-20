import numpy as np
import json
from sklearn.cross_validation import train_test_split

def f1D_task1():
    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    return f

def f2D_task2(X,Y, nb_recursive_layers=2, c1=0.3*np.pi, c2=0.4*np.pi ):
    # input layer
    X_1, X_2 = X[:,1], Y[:,2]
    # first layer
    A_1 = np.sin( c*np.multiply(X_1, X_2) )
    A_2 = np.cos( c2(X_1 + X_2) + np.pi/3 )
    # recursive layer
    for l in range(nb_recursive_layers):
        A_1 = np.multiply(3*A_1, A_2)
        A_2 = c2( np.power(X1,2) + np.power(X2,2)) - 1
    f = 2*A1 + 3*A2
    return f

def f2D_task2_draft(X,Y, nb_recursive_layers=2, c1=0.3*np.pi, c2=0.4*np.pi ):
    # input layer
    X_1, X_2 = X[:,1], Y[:,2]
    # first layer
    A_1 = np.sin( c*np.multiply(X_1, X_2) )
    A_2 = np.cos( c2(X_1 + X_2) + np.pi/3 )
    # recursive layer
    for l in range(nb_recursive_layers):
        A_1 = np.multiply(3*A_1, A_2)
        A_2 = c2( np.power(X1,2) + np.power(X2,2)) - 1
    f = 2*A1 + 3*A2
    return f

def f2D_task2(X1,X2, nb_recursive_layers=2, c1=0.3*np.pi, c2=0.4*np.pi ):


    return f

##

def generate_meshgrid_h_add():
    start_val = -1
    end_val = 1
    sqrtN = 245 #N = sqrtN*sqrtN
    N = sqrtN*sqrtN
    x_range = np.linspace(start_val, end_val, sqrtN)
    y_range = np.linspace(start_val, end_val, sqrtN)
    ## make meshgrid
    (X,Y) = np.meshgrid(x_range, y_range)
    #Z = sin(2*pi*X) + 4*(Y - 0.5).^2; %% h_add
    Z = np.sin(2*np.pi*X) + 4*np.power(Y - 0.5, 2) # h_add
    return X,Y,Z

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
            x = X[dx, dy]
            y = Y[dx, dy]
            x_data = np.array([x, y])
            y_data = Z[dx, dy]
            X_data[i,:] = x_data
            Y_data[i,:] = y_data
            i=i+1;
    return X_data, Y_data

def generate_data_task2(N_train=60000, N_cv=60000, N_test=60000, low_x_var=-2*np.pi, high_x_var=2*np.pi):

    return

def get_labels(X,Y,f):
    N_train = X.shape[0]
    for i in range(N_train):
        Y[i] = f(X[i])
    return Y

def get_labels_improved(X,f):
    N_train = X.shape[0]
    Y = np.zeros( (N_train,1) )
    for i in range(N_train):
        Y[i] = f(X[i])
    return Y

def generate_data(D=1, N_train=60000, N_cv=60000, N_test=60000, low_x_var=-2*np.pi, high_x_var=2*np.pi):
    f = f1D_task1()
    #
    low_x = low_x_var
    high_x = high_x_var
    # train

    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = get_labels(X_train, np.zeros( (N_train,D) ) , f)
    # CV
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,D)
    Y_cv = get_labels(X_cv, np.zeros( (N_cv,D) ), f)
    # test
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,D)
    Y_test = get_labels(X_test, np.zeros( (N_test,D) ), f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def generate_data(N_train_var=60000, N_cv_var=60000, N_test_var=60000, low_x_var=-2*np.pi, high_x_var=2*np.pi):
    f = f1D_task1()
    #
    low_x = low_x_var
    high_x = high_x_var
    # train
    N_train = N_train_var
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,1)
    Y_train = get_labels(X_train, np.zeros( (N_train,1) ) , f)
    # CV
    N_cv = N_cv_var
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,1)
    Y_cv = get_labels(X_cv, np.zeros( (N_cv,1) ), f)
    # test
    N_test = N_test_var
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,1)
    Y_test = get_labels(X_test, np.zeros( (N_test,1) ), f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def get_data_from_file(file_name):
    npzfile = np.load(file_name)
    # get data
    X_train = npzfile['X_train']
    Y_train = npzfile['Y_train']
    X_cv = npzfile['X_cv']
    Y_cv = npzfile['Y_cv']
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def generate_data_from_krls():
    N = 60000
    low_x =-2*np.pi
    high_x=2*np.pi
    X_train = low_x + (high_x - low_x) * np.random.rand(N,1)
    X_cv = low_x + (high_x - low_x) * np.random.rand(N,1)
    X_test = low_x + (high_x - low_x) * np.random.rand(N,1)
    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    Y_train = f(X_train)
    Y_cv = f(X_cv)
    Y_test = f(X_test)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)



def get_data(task_name):
    ## Data sets
    if task_name == 'qianli_func':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
    elif task_name == 'hrushikesh':
        with open('../hrushikesh/patient_data_X_Y.json', 'r') as f_json:
            patients_data = json.load(f_json)
        X = patients_data['1']['X']
        Y = patients_data['1']['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40)
        X_cv, X_test, Y_cv, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = ( np.array(X_train), np.array(Y_train), np.array(X_cv), np.array(Y_cv), np.array(X_test), np.array(Y_test) )
    else:
        raise ValueError('task_name: %s does not exist. Try experiment that exists'%(task_name))
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
