import sklearn
import sklearn.cluster.k_means_
import numpy as np
#from ..utils.extmath import row_norms, squared_norm
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import check_random_state

X = np.random.rand(10,3)
n_clusters = 4
random_state = None
random_state = check_random_state(random_state)
x_squared_norms = row_norms(X, squared=True)

centers = sklearn.cluster.k_means_._k_init(X, n_clusters, random_state=random_state,x_squared_norms=x_squared_norms)
print centers
