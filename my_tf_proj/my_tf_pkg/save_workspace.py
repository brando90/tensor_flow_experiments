import shelve
import numpy as np
import os

def load_results_dic(results,**kwargs):
    for key, value in kwargs.iteritems():
        results[key] = value
    return results

def numpy_save_example():
    file_name = 'some_filename'
    np.savez(file_name, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test, X_mesh_train=X_mesh_train,Y_mesh_train=Y_mesh_train,Z_mesh_train=Z_mesh_train, X_mesh_cv=X_mesh_cv,Y_mesh_cv=Y_mesh_cv,Z_mesh_cv=Z_mesh_cv, X_mesh_test=X_mesh_test,Y_mesh_test=Y_mesh_test,Z_mesh_test=Z_mesh_test)

def get_data_from_numpy_file(file_name):
    npzfile = np.load(file_name)
    # get data
    X_train = npzfile['X_train']
    Y_train = npzfile['Y_train']
    X_cv = npzfile['X_cv']
    Y_cv = npzfile['Y_cv']
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

#

def make_and_check_dir(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        # uncomment to make it raise an error when path is not a directory
        #if not os.path.isdir(path):
        #    raise
        pass

#

def save_workspace(filename, names_of_spaces_to_save, dict_of_values_to_save):
    '''
        filename = location to save workspace.
        names_of_spaces_to_save = use dir() from parent to save all variables in previous scope.
            -dir() = return the list of names in the current local scope
        dict_of_values_to_save = use globals() or locals() to save all variables.
            -globals() = Return a dictionary representing the current global symbol table.
            This is always the dictionary of the current module (inside a function or method,
            this is the module where it is defined, not the module from which it is called).
            -locals() = Update and return a dictionary representing the current local symbol table.
            Free variables are returned by locals() when it is called in function blocks, but not in class blocks.

        Example of globals and dir():
            >>> x = 3 #note variable value and name bellow
            >>> globals()
            {'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', 'x': 3, '__doc__': None, '__package__': None}
            >>> dir()
            ['__builtins__', '__doc__', '__name__', '__package__', 'x']
    '''
    print 'save_workspace'
    my_shelf = shelve.open(filename,'n') # 'n' for new
    for key in names_of_spaces_to_save:
        try:
            my_shelf[key] = dict_of_values_to_save[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            #print('ERROR shelving: {0}'.format(key))
            pass
    my_shelf.close()

def load_workspace(filename, parent_globals):
    '''
        filename = location to load workspace.
        parent_globals use globals() to load the workspace saved in filename to current scope.
    '''
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        parent_globals[key]=my_shelf[key]
    my_shelf.close()


def save_workspace_original(filename):
    my_shelf = shelve.open(filename,'n') # 'n' for new
    for key in dir():
        print key
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            #print('ERROR shelving: {0}'.format(key))
            pass
    my_shelf.close()

def load_workspace_original(filename=''):
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key]=my_shelf[key]
    my_shelf.close()
