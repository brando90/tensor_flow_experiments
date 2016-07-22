import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances
import my_tf_pkg as mtf

import matplotlib.pyplot as plt

## Data sets
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
(N_train,D) = X_train.shape
(N_test,D_out) = Y_test.shape


N = X_train.shape[0]
print N**(0.5)
print N
K = 300
indices=np.random.choice(a=N,size=K) # choose numbers from 0 to D^(1)
print 'max index',max(indices)
print 'min index',min(indices)
subsampled_data_points=X_train[indices,:] # M_sub x D
S = 10

K = mtf.get_kernel_matrix(X_train,subsampled_data_points,S)
(C,_,_,_) = np.linalg.lstsq(K,Y_train)

for i in range(1,10,K):

#m = np.linalg.lstsq(X_train,Y_train)
#print m

x = X_train
y = Y_train
y_pred = np.dot( K , C )
print (1/N)*np.linalg.norm(y_pred - y)
plt.plot(x, errors, 'o', label='Original data', markersize=1)
plt.legend()
plt.show()
