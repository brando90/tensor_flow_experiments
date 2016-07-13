import my_tf_pkg as mtf
import numpy as np

x=np.array([1,2,3])

mtf.load_workspace('a', globals())

print x #print 1,1,1 for me
