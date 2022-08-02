from typing import Dict, Union

import numpy as np
import xarray as xr

from pywatts_pipeline.core.transformer.base import BaseTransformer
from pywatts_pipeline.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray


class Merger(BaseTransformer):
    """
    The merger reduces a two-dimensional time series with multiple values per time step to a univariate time series.
    The first dimension of the input time series is the time index. The second dimension looks n-1 steps into the past
    so it contains the following values: x(t - n + 1), x(t - n + 2), .. x(t), where t is the current time index.

    :param name: The name of the module
    :type name: str
    :param method: The method indicates the merging method. Method can be 'mean', 'median', or an integer.
                   For each time step, the 'mean' method takes all values of the input time series that correspond
                   to the same time, and calculates the mean.
                   For each time step, the 'median' method takes all values of the input time series that correspond
                   to the same time and calculates the median.
                   If the method is an integer k, the k-th value in the second dimension is taken for each time step.
                   If k is negative, the k-th values in the second dimension before the last are selected.
                   If the absolute value of k is greater than the size of the horizon dimension, then the value is
                   clipped.
    :type method: Union[str,int]
    :raises WrongParameterException: If the method is not an integer, 'mean', or 'median'
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
        Set parameters for the Merger module.
        :param method:
        :type method: Union[str,int]
        :raises WrongParameterException: If the method is not an integer, 'mean', or 'median'
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
            result = self._align_temporal(horizon, x).mean(axis=0)
            return numpy_to_xarray(result[:-horizon + 1], x)
        elif self.method == "median":
            result = np.median(self._align_temporal(horizon, x), axis=0)
            return numpy_to_xarray(result[:-horizon + 1], x)
        elif isinstance(self.method, int):
            method = self.method
            if abs(self.method) >= horizon:
                method = -horizon if self.method < 0 else horizon - 1
            result = x[:, method].values
            return numpy_to_xarray(result, x)

    def _align_temporal(self, horizon, x):
        r = []
        for i in range(horizon):
            r.append(np.concatenate(
                [np.full(fill_value=np.nan, shape=(horizon - 1 - i)),
                 x.values[:, i],
                 np.full(fill_value=np.nan, shape=(i,))]))
        return np.stack(r)
