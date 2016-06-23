import numpy as np

def get_labels(X,Y,f):
    N_train = X.shape[0]
    for i in range(N_train):
        Y[i] = f(X[i])
    return Y

def generate_data(N_train_var=60000, N_cv_var=60000, N_test_var=60000, low_x_var=-2*np.pi, high_x_var=2*np.pi):
    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
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

def get_data_from_file(file_name="f_1d_cos_no_noise_data"):
    npzfile = np.load(file_name)
    # get data
    X_train = npzfile['X_train']
    Y_train = npzfile['Y_train']
    X_cv = npzfile['X_cv']
    Y_cv = npzfile['Y_cv']
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
