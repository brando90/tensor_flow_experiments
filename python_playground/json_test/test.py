import json
import numpy as np

#json.dumps( np.array([1,2,3]) )
x=json.dumps( [1,2,3] )
print x
print type(x)

#y=json.dumps(np.float32(1.0))

results = {'train_errors':[],'test_errors':[]}
results['train_errors'].append(1)
results['train_errors'].append(1)
results['train_errors'].append(1)
results['test_errors'].append(1)

print json.dumps(results)

(rnd0,rnd1,rnd2,rnd3,rnd4) = np.random.get_state()
rand_seed = (rnd0,list(rnd1),rnd2,rnd3,rnd4)
(rnd0,rnd1,rnd2,rnd3,rnd4) = rand_seed
results['rand_seed'] = rand_seed
print rand_seed
print type(rnd1[0])
rand1[0]

#results[2336278189] =  2336278189

fpath = './tmp_file'
with open(fpath,'w+') as f:
    json.dump(results,f)
