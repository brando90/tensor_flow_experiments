import json
import numpy as np

def put_numpy_seed_in_json_dic(results):
    '''
        Puts the numpy seed in the python dictionary
    '''
    (rnd0,rnd1,rnd2,rnd3,rnd4) = np.random.get_state()
    rnd1 = [int(number) for number in rnd1]
    rand_seed = (rnd0,rnd1,rnd2,rnd3,rnd4)
    results['rand_seed'] = rand_seed
    return results

def fill_results_dic_with_np_seed(np_rnd_seed, results={}):
    '''
        Puts the numpy seed in the python dictionary
    '''
    (rnd0,rnd1,rnd2,rnd3,rnd4) = np_rnd_seed
    rnd1 = [int(number) for number in rnd1]
    rand_seed = (rnd0,rnd1,rnd2,rnd3,rnd4)
    results['rand_seed'] = rand_seed
    return results

def make_numpy_seed_from(results):
    '''
        Makes a real numpy seed (tuple) from results dict
    '''
    (rnd0,rnd1,rnd2,rnd3,rnd4) = results['rand_seed']
    rnd1 = [np.uint32(number) for number in rnd1]
    rand_seed = (rnd0,rnd1,rnd2,rnd3,rnd4)
    return rand_seed

##

def example_use_lib():
    results = {'rand_seed':None}
    #Puts the numpy seed in the python dictionary
    results = mr.put_numpy_seed_in_json_dic(results)
    #Dumps the python dictionary into JSON
    fpath = './rand_seed_file'
    with open(fpath,'w+') as f:
        json.dump(results,f)
    #Load JSON into python dictionary
    with open(fpath,'r+') as f:
        results2 = json.load(f)
    #Makes a real numpy seed (tuple) from results dict
    rand_seed = mr.make_numpy_seed_from(results2)
    np.random.set_state(rand_seed)
