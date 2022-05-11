from typing import Dict, Union

import numpy as np
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions import WrongParameterException
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


class Merger(BaseTransformer):
    """
    The merger reduces a two-dimensional time series with multiple values per time step to a uni-variate time series.
    The two dimensions of the input time series is the time-index and a horizon. In the horizon dimension the next
    n values starting at the coresponding time-index are given.

    :param name: The name of the module
    :type name: str
    :param method: The method indicates the merging method. This can be mean, median, or an integer between 0 and
                   horizon -1.
                   If mean is selected. For each index in the time-series the mean of all values in the input time
                   series that corresponds to the same index are given.
                   If median is selected. For each index in the time-series the median of all values in the input time
                   series that corresponds to the same index are given.
                   If an integer n is selected. The n-th value of each sample is selected. If n is negative the n-th
                   element before the last is selected. If the absolute value of n is greater than the size of the
                   horizon dimension, then the method value is clipped.
    :type method: Union[str,int]
    :raises WrongParameterException: If method is not a non-negative integer or not 'mean' or 'median'
    """
    def __init__(self, name: str = "merger", method: Union[str, int] = "mean"):
        super().__init__(name)
        self._check_and_set_method(method)

    def _check_and_set_method(self, method):
        if method in ["mean", "median"] or isinstance(method, int):
            self.method = method
        else:
            raise WrongParameterException("The parameter method has an invalid value",
                                          "Try one of the following strings: 'mean' 'median",
                                          self)

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the Merger module.
        :return: Parameters as dict.
        :rtype: Dict[str, object]
        """
        return {
            "method" : self.method
        }

    def set_params(self, method: Union[str, int] = None):
        """
        Set parameters for the Merger module
        :param method:
        :type method: Union[str,int]
        :raises WrongParameterException: If method is not a non-negative integer or not 'mean' or 'median'
        """
        if method is not None:
            self._check_and_set_method(method)

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Merge the given time series by mean, median or selecting one specific value.
        :return: DataArray of the resulting merged time series.
        :rtype: xr.DataArray
        """
        horizon = x.values.shape[-1]
        if self.method == "mean":
            r = []
            for i in range(horizon):
                r.append(np.concatenate(
                    [np.full(fill_value=np.nan, shape=(horizon - 1 - i)),
                     x.values[:, i],
                     np.full(fill_value=np.nan, shape=(i,))]))
            result = np.stack(r).mean(axis=0)
            return numpy_to_xarray(result[:-horizon + 1], x)
        elif self.method == "median":
            r = []
            for i in range(horizon):
                r.append(np.concatenate(
                    [np.full(fill_value=np.nan, shape=(horizon - 1 - i)),
                     x.values[:, i],
                     np.full(fill_value=np.nan, shape=(i,))]))
            result = np.median(np.stack(r), axis=0)
            return numpy_to_xarray(result[:-horizon + 1], x)
        elif isinstance(self.method, int):
            method = self.method
            if abs(self.method) >= horizon:
                method = -horizon if self.method < 0 else horizon - 1
            result = x[:, method].values
            return numpy_to_xarray(result, x)
