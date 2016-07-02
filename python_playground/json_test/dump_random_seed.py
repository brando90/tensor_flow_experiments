import json
import numpy as np

def put_numpy_seed_in_json_dic(results):
    (rnd0,rnd1,rnd2,rnd3,rnd4) = np.random.get_state()
    rnd1 = [int(number) for number in rnd1]
    rand_seed = (rnd0,rnd1,rnd2,rnd3,rnd4)
    results['rand_seed'] = rand_seed
    return results

def get_numpy_seed(results):
    (rnd0,rnd1,rnd2,rnd3,rnd4) = results['rand_seed']
    rnd1 = [np.uint32(number) for number in rnd1]
    rand_seed = (rnd0,rnd1,rnd2,rnd3,rnd4)
    return rand_seed

results = {'random_seed':None}
results = put_numpy_seed_in_json_dic(results)

print np.random.rand(1)
print np.random.rand(1)
print np.random.rand(1)

fpath = './tmp_file'
with open(fpath,'w+') as f:
    json.dump(results,f)

print '...'

with open(fpath,'r+') as f:
    results2 = json.load(f)

print np.random.rand(1)

rand_seed = get_numpy_seed(results2)
np.random.set_state(rand_seed)

print np.random.rand(1)
print np.random.rand(1)
print np.random.rand(1)
