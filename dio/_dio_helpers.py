import logging
import numpy as np
import h5py as h5
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

def _loadmat_v6(filename, filedir="", paramkey="saveparams", suffix=".mat"):
    """Load .mat files lass than -v7.3"""
    import scipy.io as sio
    from pathlib import Path

    # attempt to load the file if it exists
    path = Path(Path(filedir), filename + suffix)
    if (not path.exists()) or (not path.is_file()):
        raise FileNotFoundError(f"The file specified by {str(path)} is not accesible")

    # Load the file and extract the main object
    try:
        loaded = sio.loadmat(path)
        np_params = loaded[paramkey]
    except Exception as e:
        raise Exception(f"Unable to load file with scipy package... \n{e}")

    return _getobj_recursive_v6([[np_params]])

def _getobj_recursive_v6(obj):
    """Recursive helper for goddamn .mat files"""
    if obj is str:
        return obj
    
    # unwrap object file
    try:
        obj = obj[0][0]
    except:
        obj[0][0]

    # If at a value, convert it to its native dtype and squeeze shape
    if obj.dtype.fields is None:
        dtype = obj.dtype
        return obj.astype(dtype).squeeze()
    # Else iterate through the key list and call this recursive function
    else:
        params = dict()
        keylist = list(obj.dtype.fields.keys())

        for key in keylist:
            try:
                params[key] = _getobj_recursive_v6(obj[key])
            except Exception as e:
                # Strings are weird and don't work as expected...
                params[key] = None

        return params

def _loadHDF5_recursive(obj):
    """Recursively load HDF5 groups or objects"""
    if isinstance(obj, h5.Dataset):
        obj_data = np.array(obj)
        return obj_data
    elif isinstance(obj, h5.Group):
        group_data = {}
        for key in obj.keys():
            group_data[key] = _loadHDF5_recursive(obj[key])
        return group_data