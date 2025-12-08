"""
HDF5 I/O utilities for supereeg objects.
Replaces deepdish with direct h5py usage for numpy 2.0+ compatibility.
Backwards compatible with files created by deepdish.
"""
from __future__ import print_function
try:
    import hdf5plugin  # Enables reading blosc-compressed files from deepdish
except ImportError:
    pass  # hdf5plugin is optional but needed for legacy deepdish files with blosc
import h5py
import numpy as np
import pandas as pd


def save_dict_to_hdf5(fname, data, compression='gzip'):
    """
    Save a dictionary to HDF5 file.

    Parameters
    ----------
    fname : str
        Filename to save to
    data : dict
        Dictionary of data to save
    compression : str
        Compression type ('gzip', 'lzf', 'blosc', or None).
        Note: 'blosc' is mapped to 'gzip' for h5py compatibility.
    """
    # Map deepdish compression types to h5py equivalents
    if compression == 'blosc':
        compression = 'gzip'

    with h5py.File(fname, 'w') as f:
        _recursively_save_dict(f, '/', data, compression)


def load_dict_from_hdf5(fname, group=None):
    """
    Load dictionary from HDF5 file.
    Compatible with both new h5py format and legacy deepdish format.

    Parameters
    ----------
    fname : str
        Filename to load from
    group : str, optional
        Specific group/field to load

    Returns
    -------
    dict or object
        Loaded data
    """
    with h5py.File(fname, 'r') as f:
        if group is not None:
            group_path = group.lstrip('/')
            if group_path in f:
                return _load_item(f[group_path])
            # Check if it's a root-level attribute (deepdish stores some fields as attrs)
            elif group_path in f.attrs:
                val = f.attrs[group_path]
                if isinstance(val, bytes):
                    return val.decode('utf-8')
                elif hasattr(val, 'item'):
                    # Handle numpy scalar types
                    return val.item()
                return val
            else:
                raise KeyError(f"Group '{group}' not found in file")
        return _recursively_load_dict(f, '/')


def load_slice_from_hdf5(fname, group, slices):
    """
    Load a slice of data from HDF5 file.

    Parameters
    ----------
    fname : str
        Filename to load from
    group : str
        Dataset group name
    slices : tuple
        Slice indices

    Returns
    -------
    np.ndarray
        Sliced data
    """
    with h5py.File(fname, 'r') as f:
        group_path = group.lstrip('/')
        dataset = f[group_path]

        # Handle pandas DataFrame stored in deepdish format
        if isinstance(dataset, h5py.Group):
            # This is a DataFrame or other complex type
            full_data = _load_item(dataset)
            if isinstance(full_data, pd.DataFrame):
                return full_data.iloc[slices]
            elif isinstance(full_data, np.ndarray):
                return full_data[slices]
            return full_data

        return dataset[slices]


def _recursively_save_dict(f, path, data, compression):
    """Recursively save dict items to HDF5 groups/datasets."""
    for key, item in data.items():
        full_path = f"{path}{key}" if path == '/' else f"{path}/{key}"
        _save_item(f, full_path, item, compression)


def _save_item(f, path, item, compression):
    """Save a single item to HDF5."""
    if isinstance(item, dict):
        grp = f.create_group(path)
        grp.attrs['_type'] = 'dict'
        _recursively_save_dict(f, path, item, compression)
    elif isinstance(item, pd.DataFrame):
        grp = f.create_group(path)
        grp.attrs['_type'] = 'dataframe'
        # Save values
        grp.create_dataset('values', data=item.values, compression=compression)
        # Save columns as variable-length strings
        col_data = np.array([str(c) for c in item.columns.tolist()], dtype=object)
        dt = h5py.special_dtype(vlen=str)
        grp.create_dataset('columns', data=col_data, dtype=dt)
        # Save index
        idx_data = np.array(item.index.tolist())
        grp.create_dataset('index', data=idx_data)
    elif isinstance(item, pd.Series):
        grp = f.create_group(path)
        grp.attrs['_type'] = 'series'
        grp.create_dataset('values', data=item.values, compression=compression)
        idx_data = np.array(item.index.tolist())
        grp.create_dataset('index', data=idx_data)
        if item.name is not None:
            grp.attrs['name'] = str(item.name)
    elif isinstance(item, np.ndarray):
        # Handle complex arrays
        if np.iscomplexobj(item):
            grp = f.create_group(path)
            grp.attrs['_type'] = 'complex_array'
            grp.create_dataset('real', data=item.real, compression=compression)
            grp.create_dataset('imag', data=item.imag, compression=compression)
        else:
            f.create_dataset(path, data=item, compression=compression)
    elif isinstance(item, list):
        arr = np.array(item)
        if arr.dtype == object or arr.dtype.kind == 'U':
            # Handle list of strings or mixed types
            grp = f.create_group(path)
            grp.attrs['_type'] = 'list'
            dt = h5py.special_dtype(vlen=str)
            str_arr = np.array([str(x) for x in item], dtype=object)
            grp.create_dataset('data', data=str_arr, dtype=dt)
        else:
            f.create_dataset(path, data=arr, compression=compression)
            f[path].attrs['_type'] = 'list'
    elif isinstance(item, str):
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset(path, data=item, dtype=dt)
        dset.attrs['_type'] = 'string'
    elif isinstance(item, bytes):
        f.create_dataset(path, data=item)
    elif isinstance(item, (int, float, np.integer, np.floating)):
        f.create_dataset(path, data=item)
    elif item is None:
        grp = f.create_group(path)
        grp.attrs['_type'] = 'none'
    elif isinstance(item, bool) or isinstance(item, np.bool_):
        f.create_dataset(path, data=item)
        f[path].attrs['_type'] = 'bool'
    else:
        # Try to convert to numpy array
        try:
            arr = np.array(item)
            f.create_dataset(path, data=arr, compression=compression)
        except Exception as e:
            raise TypeError(f"Cannot save type {type(item)} for path {path}: {e}")


def _load_item(item):
    """Load a single HDF5 item. Handles both new format and deepdish legacy format."""
    if isinstance(item, h5py.Dataset):
        data = item[()]
        item_type = item.attrs.get('_type', None)

        # Handle string types
        if isinstance(data, bytes):
            return data.decode('utf-8')

        if item_type == 'string':
            if isinstance(data, bytes):
                return data.decode('utf-8')
            return str(data)

        if item_type == 'list':
            return data.tolist()

        if item_type == 'bool':
            return bool(data)

        return data

    elif isinstance(item, h5py.Group):
        item_type = item.attrs.get('_type', None)

        # Check for deepdish format using TITLE attribute
        title = item.attrs.get('TITLE', None)
        if title is not None:
            if isinstance(title, bytes):
                title = title.decode('utf-8')
            elif hasattr(title, 'tolist'):
                # Handle numpy array case
                title = str(title)

            # Handle deepdish dict format
            if isinstance(title, str) and title.startswith('dict:'):
                result = {}
                # Load attributes that are values (deepdish stores small values as attrs)
                for attr_name, attr_val in item.attrs.items():
                    if attr_name not in ('CLASS', 'TITLE', 'VERSION'):
                        if isinstance(attr_val, bytes):
                            result[attr_name] = attr_val.decode('utf-8')
                        else:
                            result[attr_name] = attr_val
                # Load any groups/datasets
                for key in item.keys():
                    result[key] = _load_item(item[key])
                return result

            # Handle deepdish list format
            if isinstance(title, str) and title.startswith('list:'):
                result = []
                # Deepdish stores list items as attributes i0, i1, i2, ...
                i = 0
                while f'i{i}' in item.attrs:
                    val = item.attrs[f'i{i}']
                    if isinstance(val, bytes):
                        result.append(val.decode('utf-8'))
                    else:
                        result.append(val)
                    i += 1
                return result

        # Check for deepdish pandas_type attribute (for Series/DataFrame)
        pandas_type = item.attrs.get('pandas_type', None)
        if pandas_type is not None:
            if isinstance(pandas_type, bytes):
                pandas_type = pandas_type.decode('utf-8')

            if pandas_type == 'series':
                values = item['values'][()]
                index = item['index'][()]
                name = item.attrs.get('name', None)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                elif name == b'N.':  # deepdish uses N. for None
                    name = None
                return pd.Series(values, index=index, name=name)

            elif pandas_type == 'frame':
                # Handle deepdish DataFrame format
                # Deepdish uses block storage format
                if 'block0_values' in item:
                    # Block storage format (pandas 0.15.2 style)
                    values = item['block0_values'][()]
                    # Get column names
                    if 'axis0' in item:
                        columns_raw = item['axis0'][()]
                        columns = [c.decode('utf-8') if isinstance(c, bytes) else c
                                   for c in columns_raw]
                    elif 'block0_items' in item:
                        columns_raw = item['block0_items'][()]
                        columns = [c.decode('utf-8') if isinstance(c, bytes) else c
                                   for c in columns_raw]
                    else:
                        columns = list(range(values.shape[1]))
                    # Get index
                    if 'axis1' in item:
                        index = item['axis1'][()]
                    else:
                        index = list(range(values.shape[0]))
                    return pd.DataFrame(values, columns=columns, index=index)
                elif 'values' in item:
                    # Older simple format
                    values = item['values'][()]
                    if 'columns' in item:
                        columns = _load_item(item['columns'])
                    else:
                        columns = list(range(values.shape[1]))
                    if 'index' in item:
                        index = _load_item(item['index'])
                    else:
                        index = list(range(values.shape[0]))
                    return pd.DataFrame(values, columns=columns, index=index)
                else:
                    # Fall back to trying all keys as a dict
                    return _recursively_load_dict(item, '')

        # Handle our new format types
        if item_type == 'dataframe':
            values = item['values'][()]
            columns_raw = item['columns'][()]
            columns = []
            for c in columns_raw:
                if isinstance(c, bytes):
                    columns.append(c.decode('utf-8'))
                else:
                    columns.append(c)
            index = item['index'][()]
            return pd.DataFrame(values, columns=columns, index=index)

        elif item_type == 'series':
            values = item['values'][()]
            index = item['index'][()]
            name = item.attrs.get('name', None)
            return pd.Series(values, index=index, name=name)

        elif item_type == 'none':
            return None

        elif item_type == 'complex_array':
            real = item['real'][()]
            imag = item['imag'][()]
            # Construct complex array by direct assignment to avoid NaN issues
            # when multiplying inf values (1j * (-inf) produces nan-infj)
            result = np.empty(real.shape, dtype=np.complex128)
            result.real = real
            result.imag = imag
            return result

        elif item_type == 'list':
            data = item['data'][()]
            result = []
            for d in data:
                if isinstance(d, bytes):
                    result.append(d.decode('utf-8'))
                else:
                    result.append(d)
            return result

        elif item_type == 'dict':
            return _recursively_load_dict(item, '')

        else:
            # Empty group - could be empty dict in deepdish format
            if len(item.keys()) == 0:
                # Check if it has non-metadata attrs (deepdish stores small values as attrs)
                has_data_attrs = any(k not in ('CLASS', 'TITLE', 'VERSION')
                                    for k in item.attrs.keys())
                if has_data_attrs:
                    result = {}
                    for attr_name, attr_val in item.attrs.items():
                        if attr_name not in ('CLASS', 'TITLE', 'VERSION'):
                            if isinstance(attr_val, bytes):
                                result[attr_name] = attr_val.decode('utf-8')
                            else:
                                result[attr_name] = attr_val
                    return result
                return {}
            # Try to load as dict (for both formats)
            return _recursively_load_dict(item, '')

    return None


def _recursively_load_dict(f, path):
    """Recursively load dict from HDF5 groups."""
    result = {}
    base = f if path == '' else f[path]
    for key in base.keys():
        result[key] = _load_item(base[key])
    return result
