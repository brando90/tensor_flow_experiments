import json
import numpy as np
import my_rand_lib as mr


fpath = './rand_seed_file'
with open(fpath,'r+') as f:
    results2 = json.load(f)

rand_seed = mr.make_numpy_seed_from(results2)
np.random.set_state(rand_seed)

print np.random.rand(1)
print np.random.rand(1)
print np.random.rand(1)
