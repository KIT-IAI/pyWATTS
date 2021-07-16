from typing import Dict, List

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indices
import numpy as np

class Sampler(BaseTransformer):
    """
    This module creates samples with a size specified by sample_size. I.e., if sample_size is 24h. It creates for each
    timestamp a vector containing all values of the past 24 hours.
    E.g., this module is useful if it forecasting algorithms needs the values of the past 24 hours as input.

    :param sample_size: The offset for shifting the time series
    :type sample_size: int
    :param indices: The indices which should be shifted through time
    :type indices: List[str]

     """

    def __init__(self, sample_size: int, name: str = "SampleModule", indices: List[str] = None):
        super().__init__(name)
        if indices is None:
            indices = []
        self.sample_size = sample_size
        self.indices = indices

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of all hyperparameters/ user defined parameters

        :return: Dict with params
        :rtype: dict
        """
        return {
            "sample_size": self.sample_size,
            "indices": self.indices,
        }

    def set_params(self, sample_size: int = None, indices: List[str] = None):
        """
        Set params.

        :param sample_size: The offset for shifting the time series
        :type sample_size: int
        :param indices: The indices which should be shifted through time
        :type indices: List[str]

        """
        if sample_size:
            self.sample_size = sample_size
        if indices is not None:
            # Do not use if indices here, since this would be false if indices is empty.
            self.indices = indices

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Sample the given time series x by the lag.

        :param x: the input
        :type x: xr.DataArray
        :return: A shifted time series.
        :rtype: xr.DataArray
        """
        indices = self.indices
        if not indices:
            indices = _get_time_indices(x)
        try:
            r = [x.shift({index: i for index in indices}, fill_value=0) for i in range(0, self.sample_size)]
        except ValueError as exc:
            raise WrongParameterException(
                f"Not all indices ({indices}) are in the indices of x ({list(x.indexes.keys())}).",
                "Perhaps you set the wrong indices with set_params or during the initialization of the Sampler.",
                module=self.name) from exc
        result = xr.DataArray(np.stack(r, axis=-1), dims=(*x.dims, "horizon"), coords=x.coords)
        return result.transpose(_get_time_indices(x)[0], "horizon", ...)
