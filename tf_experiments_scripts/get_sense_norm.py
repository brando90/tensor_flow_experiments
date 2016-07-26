from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

import my_tf_pkg as mtf

def get_pairwise_norm_squared(X,Y):
    return mtf.euclidean_distances(X=X,Y=Y,squared=True)

task_name = 'hrushikesh'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(task_name)

print 'X_train.shape', X_train.shape
print 'X_cv.shape', X_cv.shape
print 'X_test.shape', X_test.shape

pairwise_norm_squared = get_pairwise_norm_squared(X=X_train,Y=X_train)
print 'min', np.amin(pairwise_norm_squared)
print 'max', np.amax(pairwise_norm_squared)
print 'mean', np.mean(pairwise_norm_squared)
print 'std', np.std(pairwise_norm_squared)
