#!/usr/bin/python

import sys
import numpy as np
import tensorflow as tf

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
print np.random.rand(1)
print tf.constant(1)

state = np.random.get_state()
#print 'rand_state', state
print np.random.rand(1)
np.random.set_state(state)
print np.random.rand(1)
