import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import my_tf_pkg as mtf

# nb_recursive_layers = 2
# file_name = 'f_2d_task2_ml_xsinlog1_x_depth_%sdata_and_mesh.npz'%(nb_recursive_layers)

print 'run task 2 data gen'
func = mtf.f2D_func_task2_2
nb_recursive_layers = 2
file_name = 'f_2d_task2_ml_xsinlog1_x_depth_%sdata_and_mesh.npz'%(nb_recursive_layers)
mtf.save_data_task2_func(func=func,file_name=file_name,nb_recursive_layers=nb_recursive_layers)
print 'data task 2 has been successfuly saved to a file'
# get data form file created
print 'get data form file created'
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name=file_name)
X,Y,Z = mtf.make_meshgrid_data_from_training_data(X_data=X_train, Y_data=Y_train)
# plot
print 'plot data'
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
#surf = ax.plot_surface(X, Y, Z, cmap=cm.BrBG)
plt.title('Original function depth = %d'%nb_recursive_layers)
plt.show()
