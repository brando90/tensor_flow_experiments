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

fpath = './tmp_file'
with open(fpath,'w+') as f:
    json.dump(results,f)
