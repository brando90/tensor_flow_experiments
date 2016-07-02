import json
import numpy as np
import dump_random_seed


fpath = './rand_seed_file'
with open(fpath,'r+') as f:
    results2 = json.load(f)

rand_seed = get_numpy_seed(results2)
np.random.set_state(rand_seed)

print np.random.rand(1)
print np.random.rand(1)
print np.random.rand(1)
