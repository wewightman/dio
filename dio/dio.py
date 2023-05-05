import logging
import numpy as np
logger = logging.getLogger(__name__)

def loadmat(filename, filedir = "", paramkey="saveparams", suffix=".mat"):
    """loads parameters from the specified .mat file"""

    from ._dio_helpers import _loadmat_v6

    # attempt to load parameters with v < 7.3 .mat file format
    try:
        logger.info(f"Attempting to open {filename+suffix}...")
        params = _loadmat_v6(filename, filedir, paramkey, suffix)
        logger.info(f"Succsess opening {filename+suffix}...")
        return params
    except Exception as e:
        logger.warn(f"Failed to open {filedir+filename+suffix} with <v7.3...")
        logger.warn(e)

    # try to load file as hdf5 format
    try:
        logger.info(f"Attempting to load {filename+suffix} with v>=7.3...")
        params = loadHDF5(filename, filedir, suffix)
        if paramkey in params.keys():
            params = params[paramkey]
        else:
            logger.warn(f"Dataset does not contain the key {paramkey}. Returning total dataset.")
        logger.info(f"Succsess opening {filename+suffix}...")
        return params
    except Exception as e:
        logger.warn(f"Unable to open {filedir+filename+suffix} with any known format...")
        raise e

def loadIQ(filename, shape, dims, dtypein=np.int32, dtypeout=np.float64, filedir="", mode='numpy',xrname="data", attr={"units":"a.u."}, isuffix="_IQreal", qsuffix="_IQimag", suffix=".bin"):
    """load IQ data into an xarray of a given shape

    Parameters:
        filename: shared name of the I and Q binary files
        shape: the shape of the binary file
        dims: the dimension labels (must have same number of labels as dimensions)
        dtypein: datatype data is saved in in the binary file
        dtyopeout: datatype data is translated to in xarray
        filedir: directory of the binary files
        xrname: name of the xarray
        attr: dictionary containing atributes of xarray (such as units)
        isuffix: the file suffix of the I binary file
        qsuffix: the file suffix of the q binary file
        suffix: suffix of the binary files

    returns:
        data: the xdata dstructure holding the I and Q components of the data

    """
    #import xarray as xr
    from pathlib import Path
    if mode == 'xarray':
        import xarray as xr

    # raise an exception if num of dimensions and dimension labels are different
    if not (len(shape) == len(dims)):
        raise Exception(f"shape (len={len(shape)}) must have same length as dims (len={len(dims)})")

    _ipath = Path(Path(filedir), filename + isuffix + suffix)
    _qpath = Path(Path(filedir), filename + qsuffix + suffix)

    # check if the binary files exist
    if not _ipath.is_file():
        raise FileNotFoundError(f"Unable to find {str(_ipath)}")

    if not _qpath.is_file():
        raise FileNotFoundError(f"Unable to find {str(_qpath)}")

    # load data from binary file
    try:
        logger.info(f"Attemping to open files ({filename})...")
        N = np.prod(shape)
        _i = np.fromfile(_ipath, count=N, dtype=dtypein)
        _q = np.fromfile(_qpath, count=N, dtype=dtypein)
        logger.info(f"Sucessfully opened binary files")
    except Exception as e:
        raise e

    # Attempt to reshape data
    try:
        logger.info(f"Reshaping data and changing format...")
        _i = _i.reshape(shape, order='F').astype(dtypeout)
        _q = _q.reshape(shape, order='F').astype(dtypeout)
        logger.info(f"Sucsessfully reshaped data and changed format...")
    except Exception as e:
        raise e

    #TODO: Convert data to xarray
    if mode == 'numpy':
        return _i, _q
    raise Exception('xarrays not yet implemented')

def loadHDF5(filename, filedir="", suffix=".h5"):
    import h5py
    from ._dio_helpers import _loadHDF5_recursive
    from pathlib import Path

    datafile = Path(Path(filedir), filename+suffix)

    if not datafile.exists():
        logger.warn(f"{str(datafile)} does not exist.")
        raise FileNotFoundError(f"{str(datafile)} does not exist.")
    
    try:
        with h5py.File(datafile) as f:
            data = {}
            for key in f.keys():
                data[key] = _loadHDF5_recursive(f[key])
    except Exception as e:
        logger.warn("Encountered error when parsing HDF5 file")
        raise e

    return data

def loadSubsetIQ(filename, shape, axis, frames, **kwargs):
    """Loads the indices along a given axis

    This function loads the specified frames along a given axis
    """
    from pathlib import Path
    ## Default values
    _default = {
        'dtypein': np.int32,
        'dtypeout': np.float64,
        'filedir':"",
        'mode':"numpy",
        'isuffix':"_IQreal",
        'qsuffix':"_IQimag",
        'suffix':".bin",
        'order':'F',
        'xarparms': {
            'attr': {'units':'a.u.'},
            'dims': []
        }
    }

    # Load default or input keywords
    _params = {}
    for k, v in _default.items():
        if k in kwargs.keys():
            _params[k] = kwargs[k]
        else:
            _params[k] = v

    # validate the desired frames
    frames = np.array(frames, dtype=int)
    axis = int(axis)

    if np.max(frames) >= shape[axis] or np.min(frames) < 0:
        raise Exception("Desired frames are out of range with given dimensions")
    
    _ifile = Path(Path(_params['filedir']), filename + _params['isuffix'] + _params['suffix'])
    _qfile = Path(Path(_params['filedir']), filename + _params['qsuffix'] + _params['suffix'])

    # Check if the file exists
    if not _ifile.exists():
        raise FileNotFoundError(f"{str(_ifile)} not found...")
    elif not _qfile.exists():
        raise FileNotFoundError(f"{str(_qfile)} not found...")

    _imap = np.memmap(_ifile, 
        dtype=_params['dtypein'], 
        shape=tuple(shape), 
        order=_params['order'], 
        mode='r')
    _qmap = np.memmap(_qfile, 
        dtype=_params['dtypein'], 
        shape=tuple(shape), 
        order=_params['order'], 
        mode='r')

    # generate slicer object to automatically slice array
    _slices = []
    for _ind, _dim in enumerate(shape):
        if not _ind == axis:
            _slices.append(slice(_dim))
        else:
            _slices.append(tuple(frames))
    _slices = tuple(_slices)

    # convert array to output datatype
    Idata = np.array(_imap[_slices], dtype=_params['dtypeout'])
    Qdata = np.array(_qmap[_slices], dtype=_params['dtypeout'])
    return Idata, Qdata