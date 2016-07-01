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

indices=np.random.choice(a=X_train.shape[0],size=dims[1]) # choose numbers from 0 to D^(1)
subsampled_data_points=X_train[indices,:] # M_sub x D
S = 100

K = mtf.get_kernel_matrix(X_train,subsampled_data_points,S)
(C,_,_,_) = np.linalg.lstsq(K,args.Y_train)

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()
