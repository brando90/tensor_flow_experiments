import json
import numpy as np
import my_rand_lib as mr

results = {'rand_seed':None}
results = mr.put_numpy_seed_in_json_dic(results)

print np.random.rand(1)
print np.random.rand(1)
print np.random.rand(1)

fpath = './rand_seed_file'
with open(fpath,'w+') as f:
    json.dump(results,f)

print '... doing other stuff'

with open(fpath,'r+') as f:
    results2 = json.load(f)

print 'other ',np.random.rand(1)
print 'other ',np.random.rand(1)
print 'other ',np.random.rand(1)

print '... done doing stuff'

rand_seed = mr.get_numpy_seed(results2)
np.random.set_state(rand_seed)

print np.random.rand(1)
print np.random.rand(1)
print np.random.rand(1)
