"""
Functionality for saving and loading model parameters, results, data, etc.

"""
import os
import json
import shutil
import logging

import numpy as np


def ow_dir_if_exists(dirname):
    """Overwrite the directory if it exists.

    Args:
        dirname (str): Name of the directory.
    Return:
        overwritten (bool): True if overwritten, else False.
    """
    try:
        shutil.rmtree(dirname)
        logging.info('overwrote old directory')
        return True
    except OSError:
        return False


def make_or_replace_dir(dirname, ow=False):
    """Same as `os.mkdir` if `ow=False`, else deletes the directory if it is
    present and recreates an empty directory in its place.

    Return:
        overwritten (bool): True if the directory was replaced, else False.
    """
    try:
        os.mkdir(dirname)
        return False
    except OSError:
        if ow:
            shutil.rmtree(dirname)
            os.mkdir(dirname)
            return True
        else:
            raise


def save_np_vars(vars, savedir, ow=False):
    """Save a dictionary of numpy variables to `savedir`.

    Args:
        vars (dict): Variables to save.
        savedir (str): Name of the directory to save the variables to.
        ow (bool): Whether or not to overwrite the savedir if it already
            exists. If False and it does exist, an OSError is raised.
    Raises:
        OSError: If `ow` is False and a directory with name `savedir` already
            exists.
    """
    logging.info('writing numpy vars to directory: %s' % savedir)
    make_or_replace_dir(savedir, ow)

    shapes = {}
    for varname, data in vars.items():
        var_file = os.path.join(savedir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape

        ## Store shape information for reloading.
        shape_file = os.path.join(savedir, 'shapes.json')
        with open(shape_file, 'w') as sfh:
            json.dump(shapes, sfh)


def load_np_vars(savedir, allow_none=True):
    """Load numpy variables saved with `save_np_vars`.

    Args:
        savedir (str): Name of directory to load vars from.
        allow_none (bool): If True (default), interpret a directory without a
            shapes.json file as containing no numpy variables and return an
            empty dict. If False, raise an IOError in this case.
    Raises:
        IOError: if `allow_none=True` and no "shapes.json" file is found in the
            `savedir`.
    Return:
        vars (dict): Dictionary of variables loaded.
    """
    logging.info('loading numpy vars from directory %s' % savedir)

    vars = {}
    shape_file = os.path.join(savedir, 'shapes.json')
    try:
        with open(shape_file, 'r') as sfh:
            shapes = json.load(sfh)
    except IOError as err:
        # if errno == 2, shapes file does not exist, assume no np vars
        if err.errno == 2 and allow_none:
            return vars
        else:
            raise

    for varname, shape in shapes.items():
        var_file = os.path.join(savedir, varname + '.txt')
        vars[varname] = np.loadtxt(var_file).reshape(shape)
        logging.debug('loaded np var %s with shape %s' % (varname, str(shape)))

    return vars


def save_model_vars(vars, savedir, ow=False):
    """Saves numpy variables using save_np_vars and saves normal vars as (type,
    data) pairs. This relies on a particular directory format. All params are
    saved as .txt files in the savedir. The shape information of numpy vars is
    saved in a file called shapes.json. The names not in this file can be
    assumed to be non-numpy vars when reading the params back in.

    Args:
        vars (dict): Variables to save.
        savedir (str): Name of the directory to save the variables to.
        ow (bool): Whether or not to overwrite the savedir if it already
            exists. If False and it does exist, an OSError is raised.
    Raises:
        OSError: If `ow` is False and a directory with name `savedir` already
            exists.
    """
    np_vars = {name: val for name, val in vars.items() if hasattr(val, 'shape')}
    save_np_vars(np_vars, savedir, ow)

    others = {name: val for name, val in vars.items() if not name in np_vars}
    for name, data in others.items():
        var_file = os.path.join(savedir, name + '.txt')
        with open(var_file, 'w') as f:
            f.write('%s,%s' % (type(data).__name__, data))


def load_model_vars(savedir):
    """Mirror function to save_model_vars.

    Returns:
        vars (dict): The variables loaded from `savedir`.
    """
    vars = load_np_vars(savedir)
    fnames = [fname for fname in os.listdir(savedir) if fname.endswith('.txt')]
    names =  [os.path.splitext(fname)[0] for fname in fnames]
    unread = [(names[i], fnames[i]) for i in range(len(names))
              if not names[i] in vars]

    for name, fname in unread:
        var_file = os.path.join(savedir, fname)
        with open(var_file) as f:
            dtype_str, data = f.read().strip().split(',')
            dtype = eval(dtype_str)
            if dtype == bool:
                vars[name] = True if data == 'True' else False
            else:
                vars[name] = dtype(data)
            logging.info('loaded var %s of type %s' % (name, dtype_str))

    return vars


def save_var_tree(tree, savedir, ow=False):
    """Save a hierarchical structure of variables specified in a dict of dicts
    to disk. This function expands upon `save_model_vars` to allow hierarchical
    directory separation of groups of variables.

    Args:
        tree (dict): A dictionary whose keys are subdirectory names and whose
            values are the dictionaries of variables to save in those
            subdirectories. This only goes one layer deep.
        savedir (str): The top-level directory to save the others within.
        ow (bool): Whether or not to overwrite the savedir if it exists.
    """
    make_or_replace_dir(savedir, ow)

    for subdirname, var_dict in tree.items():
        path = os.path.join(savedir, subdirname)
        save_model_vars(var_dict, path)


def load_var_tree(savedir):
    """Mirror function for `save_var_tree`."""
    paths = (os.path.join(savedir, name) for name in os.listdir(savedir))
    subdirs = (path for path in paths if os.path.isdir(path))
    return {os.path.basename(path): load_model_vars(path) for path in subdirs}

