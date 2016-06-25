import f_1D_data as datalib
import numpy as np

(X_train,Y_train, X_cv,Y_cv, X_test,Y_test) = datalib.generate_data()
with open('myfile.dat', 'w+') as file:
    outfile = "f_1d_cos_no_noise_data"
    np.savez(outfile, X_train=X_train, Y_train=Y_train, X_cv=X_cv, Y_cv=Y_cv, X_test=X_test, Y_test=Y_test)
    #outfile.seek(0) # Only needed here to simulate closing & reopening file
    npzfile = np.load(outfile)
    # >>> npzfile.files
    # ['y', 'x']
    # npzfile['x'] gets the np.array
    # npzfile['y'] gets the np.array
