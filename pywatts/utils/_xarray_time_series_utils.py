from typing import List, Dict

import numpy as np
import pandas as pd
import xarray as xr


def _get_time_indeces(x: Dict[str, xr.DataArray]) -> List[str]:
    indexes = []
    if isinstance(x, xr.DataArray):
        for k, v in x.indexes.items():
            if isinstance(v, pd.DatetimeIndex):
                indexes.append(k)
        return indexes
    # TODO check that all inputs have the same dimension?
    for k, v in list(x.values())[0].indexes.items():
        if isinstance(v, pd.DatetimeIndex):
            indexes.append(k)
    return indexes

def xarray_to_numpy(x: xr.Dataset):
    if x is None:
        return None
    result = None
    for data_var in x.data_vars:
        data_array = x[data_var]
        if result is not None:
            result = np.concatenate([result, data_array.values.reshape((len(data_array.values), -1))], axis=1)
        else:
            result = data_array.values.reshape((len(data_array.values), -1))
    return result

def numpy_to_xarray(x: np.ndarray, reference: xr.Dataset, name: str) -> xr.Dataset:
    coords = (
        # first dimension is number of batches. We assume that this is the time.
        ("time", list(reference.coords.values())[0].to_dataframe().index.array),
        *[(f"dim_{j}", list(range(size))) for j, size in enumerate(x.shape[1:])])

    data = {f"{name}": (tuple(map(lambda x: x[0], coords)), x),
            "time": list(reference.coords.values())[0].to_dataframe().index.array}
    return xr.Dataset(data)
