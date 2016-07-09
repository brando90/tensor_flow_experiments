import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

import my_tf_pkg as mtf


(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

replace = False # with or without replacement
stddevs = np.linspace(start=0.1, stop=4, num=50)

def learn(data, stddevs, K, replace=False):
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
    N_train = X_train.shape[0]

    for K in nb_centers_list:
        C_hat, centers, best_stddev, train_error, cv_error, test_error = mtf.get_best_shape_and_mdl(K, data, stddevs)
        if K in nb_centers_reconstruct:
            rbf_predictions_reconstruct_train.append(Y_pred)
            rbf_predictions_reconstruct_test.append(Y_pred_test)
