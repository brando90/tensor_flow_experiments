import json
import numpy as np

fpath = './tmp_file'
with open(fpath) as f:
    results = json.load(f)
    print results
    print type(results)
    print results['test_errors']
    print results['train_errors']
    print results.keys()
