import numpy as np

def get_labels(X,Y,f):
    for i in range(X)
        Y[i] = f(x[i])
    return Y

# 2*(2(cos(x)^2 - 1)^2 -1
f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
low_x = -np.pi
high_x = np.pi
# train
N_train = 60000
X_train = low_x + (high_x - low_x) * np.random.rand(N_train,1)
Y_train = get_labels(X_train,np.zeros(N_train,1),f)
# test
N_train = 60000
X_test = low_x + (high_x - low_x) * np.random.rand(N_train,1)
Y_test = get_labels(X_test,np.zeros(N_ttest,1),f)

def get_data(N_train_var=60000, N_test_var=60000,low_x_var=-np.pi, high_x_var=np.pi)
    def get_labels(X,Y,f):
        for i in range(X)
            Y[i] = f(x[i])
        return Y

    # 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    low_x = low_x_var
    high_x = high_x_var
    # train
    N_train = N_train_var
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,1)
    Y_train = get_labels(X_train,np.zeros(N_train,1),f)
    # test
    N_test = N_test_var
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,1)
    Y_test = get_labels(X_test,np.zeros(N_test,1),f)
    return (X_train, Y_train, X_test, Y_test)

#TODO load and save data sets
